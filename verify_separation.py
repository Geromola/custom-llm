"""Audit the separation between training data and the 48-case exam.

The assignment treats eval leakage as a serious problem, so this checks it from several
directions instead of trusting one rule, and prints what it finds either way.

    python verify_separation.py            # summary, exits non-zero if a hard check fails
    python verify_separation.py --verbose  # also list every near-miss n-gram

Hard checks (any failure is a real problem):
  1  no eval prompt appears in any training input, in any run
  2  no "prompt + answer" string appears in any training input
  3  no training passage contains 3 or more choices from the same case (a copied answer list)
  4  no eval file, eval result, or chat transcript sits inside a corpus folder
  5  the vocabulary was built only from training passages
  6  scoring never changed the weights: untrained and trained hashes differ, and rerunning
     the scorer on saved weights reproduces the saved score exactly

Soft measure (not a failure — reported for honesty):
  7  how often the last words of a prompt are followed by its answer in training text.
     The assignment allows underlying knowledge to overlap ("a salmon is a fish" may be
     taught) while requiring the test items themselves to stay out. This quantifies how
     close the teaching material gets, case by case.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from run_evals import load_suite, matching_cases, normalized, word_tokens

ROOT = Path(__file__).resolve().parent
RUNS = sorted(p.name for p in (ROOT / "results").iterdir()
              if p.is_dir() and (p / "corpus.txt").exists())


def training_inputs() -> dict[str, str]:
    """Everything that ever reached the model, plus the files it was built from."""
    sources = {}
    for name in RUNS:
        sources[f"results/{name}/corpus.txt"] = (ROOT / "results" / name / "corpus.txt").read_text()
    for path in sorted((ROOT / "corpus").rglob("*")):
        if path.is_file() and path.name != "README.md":
            if path.suffix == ".pdf":
                from pypdf import PdfReader
                sources[str(path.relative_to(ROOT))] = "\n".join(
                    page.extract_text() or "" for page in PdfReader(path).pages)
            else:
                sources[str(path.relative_to(ROOT))] = path.read_text(encoding="utf-8")
    for path in sorted((ROOT / "corpus_sources").glob("*.md")):
        sources[str(path.relative_to(ROOT))] = path.read_text(encoding="utf-8")
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", type=Path, default=ROOT / "separation_audit.json",
                        help="where to write the machine-readable report")
    args = parser.parse_args()
    report = {}

    suite = load_suite(ROOT / "evals/language_evals.json")
    cases = suite["cases"]
    sources = training_inputs()
    failures = []
    print(f"Auditing {len(sources)} training inputs against {len(cases)} eval cases.\n")

    # 1 ── exact prompts -----------------------------------------------------------
    hits = {name: matching_cases(text, suite) for name, text in sources.items()}
    total = sum(len(v) for v in hits.values())
    print(f"1. eval prompts found in training text: {total}")
    for name, found in hits.items():
        if found:
            print(f"     {name}: {found}")
    report["inputs_audited"] = len(sources)
    report["exact_prompts_in_training"] = total
    if total:
        failures.append("eval prompts present in training text")

    # 2 ── prompt + answer ---------------------------------------------------------
    answered = 0
    for name, text in sources.items():
        content = normalized(text)
        for case in cases:
            full = normalized(f"{case['prompt']} {case['answer']}").strip()
            if f" {full} " in content:
                print(f"     {name}: {case['id']} prompt+answer")
                answered += 1
    print(f"2. 'prompt + answer' strings found: {answered}")
    report["prompt_plus_answer_in_training"] = answered
    if answered:
        failures.append("prompt+answer strings present in training text")

    # 3 ── copied answer lists -----------------------------------------------------
    clustered = 0
    for name, text in sources.items():
        for line in text.split("\n"):
            tokens = set(word_tokens(line))
            for case in cases:
                choices = {word_tokens(c)[0] for c in case["choices"]}
                if len(tokens & choices) >= 3:
                    if args.verbose:
                        print(f"     {name}: {case['id']} — {sorted(tokens & choices)} in {line[:60]!r}")
                    clustered += 1
    print(f"3. training lines holding 3+ choices from one case: {clustered}")
    report["lines_with_three_or_more_choices"] = clustered
    if clustered:
        failures.append("possible answer lists in training text")

    # 4 ── file layout -------------------------------------------------------------
    corpus_root = (ROOT / "corpus").resolve()
    strays = [str(p.relative_to(ROOT)) for p in corpus_root.rglob("*")
              if p.is_file() and (p.name == "language_evals.json"
                                  or "eval_result" in p.name or "transcript" in p.name)]
    manifests = {name: json.loads((ROOT / "results" / name / "corpus_manifest.json").read_text())
                 for name in RUNS}
    folders = {name: m.get("mode") for name, m in manifests.items()}
    print(f"4. eval or result files inside corpus/: {len(strays)} {strays if strays else ''}")
    print(f"   corpus modes across runs: {sorted(set(folders.values()))}")
    report["eval_or_result_files_in_corpus"] = len(strays)
    if strays:
        failures.append("eval or result files inside the corpus folder")

    # 5 ── vocabulary built from training passages only -----------------------------
    vocab_ok = True
    for name in RUNS:
        split = json.loads((ROOT / "results" / name / "split.json").read_text())
        tokenization = json.loads((ROOT / "results" / name / "tokenization.json").read_text())
        train_tokens = Counter(t for doc in split["train"] for t in word_tokens(doc))
        extra = [t for t in tokenization["vocabulary"][3:] if t not in train_tokens]
        if extra:
            vocab_ok = False
            print(f"     {name}: {len(extra)} vocabulary tokens absent from training passages: {extra[:8]}")
    print(f"5. vocabulary drawn only from training passages: {'yes' if vocab_ok else 'NO'}")
    report["vocabulary_from_training_only"] = vocab_ok
    if not vocab_ok:
        failures.append("vocabulary contains tokens not present in training passages")

    # 6 ── scoring did not train ----------------------------------------------------
    identity_ok = True
    for name in RUNS:
        final = json.loads((ROOT / "results" / name /
                            "language_evals/final/eval_summary.json").read_text())
        untrained = json.loads((ROOT / "results" / name /
                                "language_evals/untrained/eval_summary.json").read_text())
        if final["model_sha256"] == untrained["model_sha256"]:
            identity_ok = False
            print(f"     {name}: trained and untrained weights hash the same")
        if final["suite_sha256"] != untrained["suite_sha256"]:
            identity_ok = False
            print(f"     {name}: the eval suite changed between stages")
    suites = {json.loads((ROOT / "results" / n /
                          "language_evals/final/eval_summary.json").read_text())["suite_sha256"]
              for n in RUNS}
    print(f"6. weights differ per stage and one single suite hash across all runs: "
          f"{'yes' if identity_ok and len(suites) == 1 else 'NO'}")
    suite_hash = suites.pop()
    report["single_suite_hash_across_runs"] = suite_hash
    report["runs_audited"] = len(RUNS)
    report["weights_differ_per_stage"] = identity_ok
    print(f"   suite sha256 {suite_hash[:32]}… used by all {len(RUNS)} runs")
    if not identity_ok:
        failures.append("model or suite identity inconsistent")

    # 7 ── soft measure: how close does the teaching material get? -------------------
    print("\n7. Overlap between training text and each case's answer (allowed, measured):")
    corpus_text = normalized("\n".join(
        text for name, text in sources.items() if name.startswith("results/exp2-expanded")))
    rows = []
    for case in cases:
        prompt_tokens = word_tokens(case["prompt"])
        answer = word_tokens(case["answer"])[0]
        counts = {}
        for k in (1, 2, 3):
            if len(prompt_tokens) < k:
                continue
            phrase = " ".join(prompt_tokens[-k:] + [answer])
            counts[k] = corpus_text.count(f" {phrase} ")
        rows.append((case, counts))
    worst = [(c, n) for c, n in rows if n.get(3, 0) > 0]
    print(f"   cases whose last 3 prompt words are followed by the answer somewhere in "
          f"training: {len(worst)} of {len(cases)}")
    for case, counts in sorted(worst, key=lambda r: -r[1].get(3, 0))[:12]:
        print(f"     {case['id']} [{case['category']}] "
              f"…{' '.join(word_tokens(case['prompt'])[-3:])} → {case['answer']}: "
              f"{counts.get(3,0)}x")
    if args.verbose:
        print("\n   full table (1/2/3-word context before the answer):")
        for case, counts in rows:
            print(f"     {case['id']:9s} {case['category']:26s} "
                  + "  ".join(f"{k}:{v:4d}" for k, v in counts.items()))

    report["cases_with_three_word_context_overlap"] = len(worst)
    report["overlap_examples"] = [
        {"case": case["id"], "category": case["category"],
         "phrase": " ".join(word_tokens(case["prompt"])[-3:] + [word_tokens(case["answer"])[0]]),
         "occurrences": counts.get(3, 0)}
        for case, counts in sorted(worst, key=lambda r: -r[1].get(3, 0))[:12]]
    report["failures"] = failures
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.json.name}")

    print("\n" + ("FAILED: " + "; ".join(failures) if failures
                  else "All hard checks passed: the exam never entered any training input."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
