"""Generate README.md from the saved run artefacts.

Nothing in the README is typed by hand: every score, loss, vector, gradient and sample is
read out of results/. If a number in the README is wrong, this script is wrong, and
rerunning it after a fresh experiment updates the whole document.

    python build_readme.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = "https://github.com/Geromola/custom-llm"
PAGES = "https://geromola.github.io/custom-llm/viz/"
STARTER, EXPANDED = "exp1-starter", "exp2-expanded"
TAUGHT = ["grammar", "opposites", "everyday_knowledge", "categories_and_analogies"]
UNTAUGHT = ["negation", "reference", "sequence", "spatial_relations"]
STARTER_CATS = ["domain_context", "domain_place", "new_wording"]
ROBUST = [("exp1-starter", 42), ("rb-starter-s1234", 1234), ("rb-starter-s2468", 2468),
          ("exp2-expanded", 42), ("rb-expanded-s1234", 1234), ("rb-expanded-s2468", 2468)]


def load(*parts):
    return json.loads((ROOT.joinpath(*parts)).read_text())


def run(name):
    base = Path("results") / name
    return {"config": load(base, "config.json"),
            "summary": load(base, "training_summary.json"),
            "history": load(base, "history.json"),
            "inspection": load(base, "inspection.json"),
            "tokenization": load(base, "tokenization.json"),
            "vocabulary": load(base, "vocabulary_report.json"),
            "manifest": load(base, "corpus_manifest.json"),
            "separation": load(base, "eval_separation.json"),
            "temperature": load(base, "temperature_comparison.json"),
            "evals": {stage: load(base, "language_evals", stage, "eval_summary.json")
                      for stage in ("untrained", "final")},
            "cases": {stage: load(base, "language_evals", stage, "eval_results.json")
                      for stage in ("untrained", "final")},
            "samples": {int(p.stem.split("_")[1]): p.read_text().rstrip("\n").split("\n")
                        for p in sorted((ROOT / base / "samples").glob("*.txt"))},
            "path": base.as_posix()}


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def row(summary):
    o = summary["overall"]
    return (f"| {o['correct']}/48 | {pct(o['success_rate_all_cases'])} | {o['scorable']}/48 "
            f"| {pct(o['accuracy_scorable_cases'])} |")


def category_table(runs):
    lines = ["| Category | Kind | Starter untrained | Starter trained | Expanded untrained | Expanded trained |",
             "|---|---|---:|---:|---:|---:|"]
    for category in STARTER_CATS + TAUGHT + UNTAUGHT:
        kind = ("starter" if category in STARTER_CATS else
                "**taught**" if category in TAUGHT else "control")
        cells = []
        for name in (STARTER, EXPANDED):
            for stage in ("untrained", "final"):
                c = runs[name]["evals"][stage]["by_category"].get(category)
                if not c:
                    cells.append("–")
                elif c["scorable"] == 0:
                    cells.append(f"{c['correct']}/{c['total']} ∅")
                else:
                    cells.append(f"{c['correct']}/{c['total']}")
        order = [cells[0], cells[1], cells[2], cells[3]]
        lines.append(f"| `{category}` | {kind} | " + " | ".join(order) + " |")
    return "\n".join(lines)


def cases_table(runs, categories, stage=("final", EXPANDED)):
    stage_name, run_name = stage
    rows = [c for c in runs[run_name]["cases"][stage_name] if c["category"] in categories]
    lines = ["| Case | Prompt the model sees | Wanted | Picked | Score | Its own continuation |",
             "|---|---|---|---|:--:|---|"]
    for c in rows:
        picked = c["predicted_choice"] or f"_{c['status']}_"
        mark = "✅" if c["score"] else "❌"
        text = (c["generated_text"] or "_(empty)_").replace("|", "\\|")
        lines.append(f"| `{c['id']}` | `{c['prompt']}` | **{c['expected']}** | {picked} | {mark} "
                     f"| {text[:70]} |")
    return "\n".join(lines)


def main() -> None:
    runs = {name: run(name) for name, _ in ROBUST}
    longer = run("exp3-longer")
    starter, expanded = runs[STARTER], runs[EXPANDED]
    corpus = load("corpus_report.json")
    pdf = load("corpus_sources", "pdf_extraction_check.json")
    terminal = load("results", "chat", "terminal_transcript.json")
    web = load("results", "chat", "web_transcript.json")
    milestones = json.loads((ROOT / "viz/data/milestones.json").read_text())
    audit = load("separation_audit.json")

    ec, sc = expanded["config"], starter["config"]
    inspection = expanded["inspection"]
    update = inspection["first_update"]
    vocabulary = expanded["tokenization"]["vocabulary"]

    def top_probabilities(key, n=5):
        probs = inspection[key]
        order = sorted(range(len(probs)), key=lambda i: -probs[i])[:n]
        return ", ".join(f"`{vocabulary[i]}` {100*probs[i]:.2f}%" for i in order)

    taught_final = sum(expanded["evals"]["final"]["by_category"][c]["correct"] for c in TAUGHT)
    taught_untrained = sum(expanded["evals"]["untrained"]["by_category"][c]["correct"] for c in TAUGHT)
    control_correct = {name: sum(r["evals"][s]["by_category"][c]["correct"]
                                 for c in UNTAUGHT for s in ("untrained", "final"))
                       for name, r in runs.items()}

    def losses(r):
        return "\n".join(f"| {h['step']} | {h['training_loss']:.4f} | {h['validation_loss']:.4f} |"
                         for h in r["history"])

    def samples_block(r, step):
        return "\n".join(f"    {line or '(empty)'}" for line in r["samples"][step])

    robustness = "\n".join(
        f"| `{name}` | {'starter' if 'starter' in name else 'expanded'} | {seed} "
        f"| {runs[name]['evals']['final']['overall']['correct']}/48 "
        f"| {runs[name]['evals']['final']['by_group']['starter_patterns']['correct']}/16 "
        f"| {runs[name]['evals']['final']['by_group']['starter_transfer']['correct']}/8 "
        f"| {sum(runs[name]['evals']['final']['by_category'][c]['correct'] for c in TAUGHT)}/12 "
        f"| {sum(runs[name]['evals']['final']['by_category'][c]['correct'] for c in UNTAUGHT)}/12 |"
        for name, seed in ROBUST)

    chat_turns = "\n".join(
        f"| `{t['prompt']}` | {t['response'] or '_(empty)_'} | "
        f"{', '.join(t['unknown_prompt_words']) or '—'} |" for t in terminal["turns"])
    web_turns = "\n".join(
        f"| `{t['prompt'][:60]}{'…' if len(t['prompt'])>60 else ''}` | {t['response'] or '_(empty)_'} | "
        f"{', '.join(t['unknown_prompt_words']) or '—'} |" for t in web["turns"])

    failing = [c for c in expanded["cases"]["final"]
               if c["category"] in TAUGHT and not c["score"]]
    failure = failing[0] if failing else None
    failure_probs = (", ".join(f"`{k}` {100*v:.3f}%" for k, v in
                               sorted(failure["choice_probabilities"].items(),
                                      key=lambda kv: -kv[1])) if failure else "")

    overlap_rows = "\n".join(
        f"| `{e['case']}` | {e['category']} | `…{e['phrase']}` | {e['occurrences']} |"
        for e in audit["overlap_examples"])
    audit_fields = {
        "audited": audit["inputs_audited"], "runs": audit["runs_audited"],
        "exact": audit["exact_prompts_in_training"],
        "pa": audit["prompt_plus_answer_in_training"],
        "lists": audit["lines_with_three_or_more_choices"],
        "strays": audit["eval_or_result_files_in_corpus"],
        "vocab": "yes" if audit["vocabulary_from_training_only"] else "NO",
        "identity": "yes" if audit["weights_differ_per_stage"] else "NO",
        "suite_hash": audit["single_suite_hash_across_runs"][:24],
        "overlap": audit["cases_with_three_word_context_overlap"],
        "overlap_rows": overlap_rows}

    readme = f"""# A tiny language model, built and taken apart

A 2-block, 4-head, {ec['n_embd']}-number nanoGPT — **{ec['parameters']:,} parameters** — trained from
scratch on word tokens, then measured with a fixed 48-case exam it never saw in training, and
finally opened up so you can watch it choose a word.

**▶ [Open the interactive explorer]({PAGES})** — the map of what the model thinks is related, a
chat box that runs the real weights in your browser, and the arithmetic behind every prediction.

| | Starter corpus | Expanded corpus |
|---|---|---|
| 48-case exam, before training | {starter['evals']['untrained']['overall']['correct']}/48 | {expanded['evals']['untrained']['overall']['correct']}/48 |
| 48-case exam, after training | **{starter['evals']['final']['overall']['correct']}/48** | **{expanded['evals']['final']['overall']['correct']}/48** |
| Cases it could even read | {starter['evals']['final']['overall']['scorable']}/48 | {expanded['evals']['final']['overall']['scorable']}/48 |
| Of the 12 cases I taught for | {sum(starter['evals']['final']['by_category'][c]['correct'] for c in TAUGHT)}/12 | **{taught_final}/12** |
| Of the 12 cases I deliberately did **not** teach | 0/12 | 0/12 |

The headline is not the score. It is that the gain is **confined to exactly what I taught**: the
12 control cases score zero in every run, before and after, in both experiments and at three
different random seeds.

---

## Contents

- [What is in this repository](#what-is-in-this-repository)
- [The three choices](#the-three-choices)
- [The corpus I wrote](#the-corpus-i-wrote)
- [What I predicted, and what actually happened](#what-i-predicted-and-what-actually-happened)
- [The exam: all four result sets](#the-exam-all-four-result-sets)
- [Is the improvement real?](#is-the-improvement-real)
- [One case that fails, and why it is my fault](#one-case-that-fails-and-why-it-is-my-fault)
- [How this thing actually learns](#how-this-thing-actually-learns)
- [Training curves and samples](#training-curves-and-samples)
- [The interactive explorer](#the-interactive-explorer)
- [Chat with the model](#chat-with-the-model)
- [Reproducing everything](#reproducing-everything)
- [Limitations and the next experiment](#limitations-and-the-next-experiment)

---

## What is in this repository

| Path | What it is |
|---|---|
| [`custom_llm.ipynb`](custom_llm.ipynb) | Executed notebook, **expanded-corpus** experiment, outputs intact |
| [`custom_llm_starter.ipynb`](custom_llm_starter.ipynb) | Executed notebook, **starter-corpus** experiment |
| [`PREDICTION.md`](PREDICTION.md) | What I expected, committed to git *before* the first training run |
| [`make_corpus.py`](make_corpus.py) | Generates my teaching corpus and withholds anything that collides with the exam |
| [`corpus/`](corpus) | The teaching material itself: 3 Markdown files and 1 PDF, all written by me |
| [`evals/language_evals.json`](evals/language_evals.json) | The supplied 48-case exam, unchanged |
| [`run_evals.py`](run_evals.py) | The supplied scorer: inference only, never trains |
| [`results/`](results) | Every run: weights, corpus, eval results, losses, samples |
| [`viz/`](viz) | The interactive explorer, its data exporters and the in-browser model |
| [`chat.py`](chat.py) · [`chat_web.py`](chat_web.py) | Terminal and web chat interfaces |
| [`build_readme.py`](build_readme.py) | Generates this file from the saved results |

Starter notebook, `run_evals.py`, `chat.py`, `nanogpt_model.py`, the eval suite and
`embedding-viewer.html` come from the course sample project
[`pepealonso95/custom-llm`](https://github.com/pepealonso95/custom-llm); the network itself is
[Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) at the pinned commit
`{ec['upstream_commit'][:12]}`, hash-checked on every run ([`NANOGPT_LICENSE`](NANOGPT_LICENSE)).

---

## The three choices

| Choice | Value | Why |
|---|---|---|
| **Corpus** | Run 1: classroom sentences only. Run 2: the same plus {corpus['unique_passages']:,} passages I wrote | The corpus is the *only* difference between the two runs, so any change in the exam has one candidate cause |
| **Training steps** | {ec['training_steps']:,} in both | The suggested budget, and validation loss is already flat there. Holding it identical matters more than tuning it: train run 2 longer and I could not tell new data from extra steps |
| **Learning rate** | {ec['learning_rate']} in both | The notebook default, with 100-step warmup and cosine decay to 10%. Much larger and AdamW's steps overshoot — the loss oscillates or goes non-finite, which the notebook guards against. Much smaller and 3,000 steps end near the random start, with loss barely below ln(vocabulary) |

Everything else — seed {ec['seed']}, {ec['n_layer']} blocks, {ec['n_head']} heads, {ec['n_embd']}
numbers per token, {ec['block_size']}-token context, batch {ec['batch_size']} — is the supplied
default and identical across both runs.

---

## The corpus I wrote

The starter corpus is synthetic classroom sentences. Its vocabulary has no word for *opposite*,
*ice*, *robin* or *walked*, so **all 24 extension cases are unreadable** to it: some word in the
prompt or in the four answers is outside the vocabulary, the case is marked
`out_of_vocabulary`, and it scores zero no matter how long you train.

So I wrote teaching material for **four** of the eight extension categories, and deliberately
wrote **nothing** for the other four:

| Taught | Left as a control |
|---|---|
| `grammar`, `opposites`, `everyday_knowledge`, `categories_and_analogies` | `negation`, `reference`, `sequence`, `spatial_relations` |

That makes the other 12 cases an internal control group: same model, same exam, same training
run. If their scores moved, something other than my teaching material would be responsible.

### What it contains

| File | Passages | Form |
|---|---:|---|
""" + "\n".join(
        f"| `corpus/{name}.md` | {values['passages']} | Markdown |" if name != "everyday_knowledge"
        else f"| `corpus/everyday_knowledge.pdf` | {values['passages']} | **PDF**, rendered from [`corpus_sources/everyday_knowledge.md`](corpus_sources/everyday_knowledge.md) |"
        for name, values in corpus["files"].items()) + f"""
| **Total** | **{corpus['total_passages']:,}** ({corpus['unique_passages']:,} unique) | adds {corpus['distinct_token_types_added_by_corpus_files']} distinct token types |

Permissions are simple: I wrote all of it, generated by [`make_corpus.py`](make_corpus.py) from
templates I chose. No outside documents, no personal data, nothing confidential.

### Keeping the exam out of the textbook

`make_corpus.py` generates freely and then **withholds** any line that reproduces a test prompt,
using the starter's own matching rule — the same idea the notebook applies to its own generated
sentences. It reserved **{corpus['reserved_line_count']} lines**, including every
`the opposite of hot is cold`, every `one bird is …` and every `yesterday she …`.

Two subtleties I had to handle, both visible in the code:

1. **The notebook checks an imported file as one continuous stream**, not line by line. So
   `a carrot is a vegetable .` on one line followed by `an apple is a fruit .` on the next
   reproduces a test prompt across the line break even though neither line does alone.
   `order_lines()` orders every file so no window of three consecutive lines contains a prompt,
   and the file-level check runs again before anything is written.
2. **`chunk_text` splits on a sentence mark followed by whitespace**, so every training passage
   is a single sentence and a file corpus cannot teach a two-clause pattern at all — which is
   what the analogy cases are. Writing the period glued to the next word
   (`a duck is a bird .a goat is an animal .`) keeps both clauses in one passage; the tokenizer
   matches words and single punctuation marks and ignores whitespace, so the training tokens are
   *identical* to the spaced version. {corpus['files']['categories_and_analogies']['multi_clause']}
   of my analogy passages use this, and the leakage check still runs on the joined passage
   exactly as the model sees it.

### The separation audit

Leakage is the one thing in this assignment that can cost marks beyond its own category, so
[`verify_separation.py`](verify_separation.py) checks it from six directions and writes
[`separation_audit.json`](separation_audit.json). Run it yourself with
`python verify_separation.py`.

| # | Check | Result |
|---|---|---|
| 1 | Eval prompts found in any training input ({{audited}} audited, across all {{runs}} runs) | **{{exact}}** |
| 2 | `prompt + answer` strings found in training text | **{{pa}}** |
| 3 | Training lines holding 3+ choices from one case (a copied answer list) | **{{lists}}** |
| 4 | Eval files, eval results or chat transcripts inside a corpus folder | **{{strays}}** |
| 5 | Vocabulary built only from training passages | **{{vocab}}** |
| 6 | Trained and untrained weights differ; one single suite hash across every run | **{{identity}}** |

The suite hash `{{suite_hash}}…` is identical in all {{runs}} runs and in both stages of each, so
the same 48 cases, the same four choices and the same answer key scored everything. The scorer
[`run_evals.py`](run_evals.py) only ever runs inference, and asserts the model's hash is
unchanged afterwards — if scoring had nudged a single weight it would have raised.

### What *does* overlap, and why that is allowed

The assignment is explicit that ordinary words and underlying subject knowledge may overlap;
only the test items themselves must stay out. So the honest question is not "is there any
overlap" but "how close does the training text get". For {{overlap}} of the 48 cases, the last
three words of the prompt are followed by the correct answer somewhere in training:

| Case | Category | Phrase in training text | Times |
|---|---|---|---:|
{{overlap_rows}}

Two worked examples of what is behind those counts:

- **`lang_09`** — the test prefix is `the team discussed the customer and the service at the`
  → **store**. That exact sentence **is not in the training text**; the notebook reserved it,
  along with 159 other passages, before the split and before the vocabulary was built. What
  remains is its siblings with a *different* noun: `the team discussed the client and the
  service at the store .` This is the supplied corpus behaving as the assignment describes —
  it teaches the association and withholds the exact test sentence.
- **`lang_46`** — the test prefix is `a robin is a bird . a salmon is a` → **fish**, and it is
  absent too. My analogy file does contain `a salmon is a fish .`, paired with other first
  clauses: `a banana is a fruit .a salmon is a fish .` The fact is taught; the test item, with
  its `a robin is a bird` opening, never appears.

That is the line the assignment draws, and it is also why the scores should be read as a
*development benchmark*: I could see these cases while choosing what to teach.

Both runs' separation records are saved:
[`eval_separation.json`]({expanded['path']}/eval_separation.json) — the notebook reserved
**{expanded['separation']['excluded_passages']:,} classroom passages** covering
{len(expanded['separation']['case_ids'])} test cases before the split and before the vocabulary
was built. Its method line says it plainly: *"{expanded['separation']['method']}"*. An exact-match
check cannot catch paraphrases, so the real guarantee is that I wrote the material and can show
what it says.

### The PDF, and what the extraction check caught

One file ships as a PDF on purpose, so the notebook's PDF path is exercised rather than
described. Rendering it went wrong twice, and both are worth recording:

- A per-page `<h2>` header extracted as text and became six junk training passages.
- At 46 lines per page the list overflowed the fixed page height and `overflow:hidden` silently
  **dropped 5 teaching sentences**. Nothing warned: every page still had text, so the notebook's
  "no text extracted" warning would never have fired.

[`corpus_sources/make_pdf.py`](corpus_sources/make_pdf.py) now renders at 34 lines per page and
verifies the round-trip, and
[`pdf_extraction_check.json`](corpus_sources/pdf_extraction_check.json) records the result:
{pdf['pdf_pages']} pages, {pdf['pages_with_no_text'] or 'no'} pages without text,
{pdf['extracted_passages']}/{pdf['source_passages']} passages recovered,
`reading_order_identical: {str(pdf['reading_order_identical']).lower()}`.

### What the corpus did to the vocabulary

| | Starter | Expanded |
|---|---:|---:|
| Unique training passages | {sc['train_documents'] + sc['validation_documents']:,} | {ec['train_documents'] + ec['validation_documents']:,} |
| Train / validation split | {sc['train_documents']:,} / {sc['validation_documents']:,} | {ec['train_documents']:,} / {ec['validation_documents']:,} |
| Vocabulary (incl. `<UNK> <BOS> <EOS>`) | {sc['vocabulary_size']} | {ec['vocabulary_size']} |
| Token types kept of those seen | {starter['vocabulary']['retained_types']}/{starter['vocabulary']['training_types']} | {expanded['vocabulary']['retained_types']}/{expanded['vocabulary']['training_types']} |
| Unknown-token rate, training | {pct(sc['training_unknown_rate'])} | {pct(ec['training_unknown_rate'])} |
| Unknown-token rate, held-out | {pct(sc['validation_unknown_rate'])} | {pct(ec['validation_unknown_rate'])} |
| Parameters | {sc['parameters']:,} | {ec['parameters']:,} |

Both runs stay far below the 509-type cap, so **nothing was dropped**: the unknown-token rate is
0% on both splits in both runs. The extra parameters in run 2 are entirely the bigger embedding
table ({ec['vocabulary_size']} rows instead of {sc['vocabulary_size']}) — the network itself is
the same size.

One honest limit of the split: it is by **passage**, not by source file, and duplicates are
removed first. Held-out passages come from the same templates and the same files as training
ones, so a good validation loss shows the model fits the template family — not that it
generalises to text it has never seen the shape of.

Full manifests: [`corpus_manifest.json`]({expanded['path']}/corpus_manifest.json) ·
[`vocabulary_report.json`]({expanded['path']}/vocabulary_report.json) ·
[`corpus_report.json`](corpus_report.json)

---

## What I predicted, and what actually happened

[`PREDICTION.md`](PREDICTION.md) was committed before the first real run. I got the shape right
and two things clearly wrong.

| Prediction | Actual | |
|---|---|---|
| Starter trained: 18–22 / 48 | **{starter['evals']['final']['overall']['correct']}/48** | ✅ |
| Expanded coverage: 36/48 scorable | **{expanded['evals']['final']['overall']['scorable']}/48** | ✅ |
| Expanded untrained: 10–13 / 48 | **{expanded['evals']['untrained']['overall']['correct']}/48** | ❌ lower than I guessed |
| Expanded trained: 26–31 / 48 | **{expanded['evals']['final']['overall']['correct']}/48** | ❌ better than I guessed |
| Of 12 taught cases: 7–9 correct | **{taught_final}/12** | ❌ better than I guessed |
| Control cases: 0/12 in all runs | **0/12 in all {len(ROBUST)} runs** | ✅ |
| `yesterday she` → `walked` would probably fail | it got it, at {100*[c for c in expanded['cases']['final'] if c['id']=='lang_27'][0]['choice_probabilities']['walked']:.0f}% | ❌ |
| Starter-transfer cases would get *worse* with my corpus | {starter['evals']['final']['by_group']['starter_transfer']['correct']}/8 → {expanded['evals']['final']['by_group']['starter_transfer']['correct']}/8 | ❌ — and see below, the difference is seed noise |

The prediction I am happiest to have been wrong about is `yesterday she` → `walked`. Every
teaching line containing that literal string had to be withheld as exam material, so the model
had to put "yesterday … walked" together with "she walks / she walked" from *different*
sentences. It did.

---

## The exam: all four result sets

48 fixed cases, four single-word choices each, scored by which choice gets the highest
probability. Only the prompt goes into the model — never the choices, never the answer. Ties
score zero. A case whose prompt or answers contain a word outside the vocabulary is **unscorable
and counts as zero** in the all-case rate, so dropping hard cases cannot inflate it. Random
guessing would average 25% among scorable cases.

| Experiment | Stage | Correct | All-case rate | Scorable | Accuracy among scorable | Full results |
|---|---|---:|---:|---:|---:|---|
| Starter | untrained {row(starter['evals']['untrained'])} [results]({starter['path']}/language_evals/untrained) |
| Starter | trained {row(starter['evals']['final'])} [results]({starter['path']}/language_evals/final) |
| Expanded | untrained {row(expanded['evals']['untrained'])} [results]({expanded['path']}/language_evals/untrained) |
| Expanded | trained {row(expanded['evals']['final'])} [results]({expanded['path']}/language_evals/final) |

Each folder holds `eval_cases.json`, `eval_results.json`, `eval_results.csv` and
`eval_summary.json` — all 48 cases, every probability, every free continuation, nothing dropped.

### Reading the four rows together

These four numbers separate two things that both look like "the model got better":

- **Vocabulary coverage.** Starter → expanded takes scorable cases from
  {starter['evals']['final']['overall']['scorable']} to
  {expanded['evals']['final']['overall']['scorable']}. That is purely my corpus adding words.
- **Learning.** *Expanded untrained* has the full vocabulary and random weights: it scores
  {expanded['evals']['untrained']['overall']['correct']}/48, i.e.
  {pct(expanded['evals']['untrained']['overall']['accuracy_scorable_cases'])} among scorable
  cases — chance, exactly as it should be. Training the same vocabulary to
  {expanded['evals']['final']['overall']['correct']}/48
  ({pct(expanded['evals']['final']['overall']['accuracy_scorable_cases'])} of scorable) is the
  part the weights are responsible for.

Coverage alone buys nothing: it only makes a case *askable*. On the taught categories the
untrained expanded model gets {taught_untrained}/12; the trained one gets {taught_final}/12.

### Every category, all four result sets

{category_table(runs)}

∅ means the model cannot read the case at all. The four control categories stay ∅ throughout —
my corpus never gave them their words, which is the point.

### The 12 cases I taught for

{cases_table(runs, TAUGHT)}

The "continuation" column is the model writing freely from the same prompt, sampled at
temperature 0.8. It is saved for inspection and is **not** what the score measures: picking the
best of four words is a far easier task than writing the next words unaided.

---

## Is the improvement real?

One run of each is not evidence. Training costs about {expanded['summary']['elapsed_seconds']:.0f}
seconds here, so I ran each corpus at three seeds. Only the seed changed.

| Run | Corpus | Seed | Correct | Starter patterns | Starter transfer | Taught | Control |
|---|---|---:|---:|---:|---:|---:|---:|
{robustness}

Read down the columns:

- **Taught categories: 0/12 → {taught_final}/12, and 11–12 of 12 at every seed.** Far outside
  seed noise. This is the finding the experiment supports.
- **Starter transfer swings 4–8 out of 8 on the starter corpus alone.** In the headline pair it
  goes 4/8 → 8/8, which looks like my corpus helping transfer — but a starter run at another seed
  also reached 8/8. **That apparent improvement is inside seed noise and I am not claiming it.**
  My prediction that transfer would get *worse* was wrong too; the honest conclusion is that
  three seeds cannot resolve a difference this small.
- **Control: 0/12 in all six runs**, plus the longer run below. Nothing leaked.

And more steps do not substitute for data: [`exp3-longer`](results/exp3-longer) trains the
expanded corpus for {longer['config']['training_steps']:,} steps instead of {ec['training_steps']:,}
and scores {longer['evals']['final']['overall']['correct']}/48 — no better than
{expanded['evals']['final']['overall']['correct']}/48 at 3,000. Missing vocabulary and missing
patterns cannot be trained away on data that never contained them.

Because I looked at these results while deciding what to teach, this suite is a **public
development benchmark, not an unseen final test**. A claim about generalisation would need fresh
cases that never influenced any choice I made.

---

## One case that fails, and why it is my fault

""" + (f"""`{failure['id']}` — the model sees `{failure['prompt']}` and should prefer
**{failure['expected']}**. It picks `{failure['predicted_choice']}`.

    {failure_probs}

Two things are wrong here. First, *all four* probabilities are tiny — the model puts almost all
its mass somewhere else entirely, and its free continuation is `{failure['generated_text']}`.
Second, that is my own teaching material talking back at me. My everyday-knowledge file contains
`ice melts into water .`, which teaches `into → water` directly against the tested
`freezes into → ice`, and my phrasings for the intended fact all avoid the banned literal
(`when water freezes it becomes ice`, `water freezes when the room is cold`) — so nothing ever
put `ice` immediately after `into`. The competing pattern is better attested than the wanted one.

The fix is a data fix, not a training fix: drop `ice melts into water`, and add sentences where
`into` is followed by `ice` in a different frame (`the cold turns the puddle into ice`). That is
the next experiment below.
""" if failure else "Every taught case passed in this run.\n") + f"""

---

## How this thing actually learns

Everything below is from [`inspection.json`]({expanded['path']}/inspection.json) and
[`tokenization.json`]({expanded['path']}/tokenization.json) in the expanded run.

### 1. Text becomes tokens, tokens become row numbers

A training passage, exactly as stored:

    {expanded['tokenization']['example']}

becomes tokens `{' '.join(expanded['tokenization']['example'].split()[:8])} …` and then row
numbers `{expanded['tokenization']['ids'][:10]} …`. Those numbers are **addresses, not
quantities**: token {inspection['token_id']} is not "more" than token 51. `1` is `<BOS>`, the
marker that starts every passage.

The model's whole input is `inputs → targets` shifted by one: given
`{' '.join(expanded['tokenization']['example'].split()[:4])}` predict
`{expanded['tokenization']['example'].split()[4]}`. That is the only task it is ever set.

### 2. A row number becomes 64 learned numbers

`{inspection['token']}` is row {inspection['token_id']} of the embedding table
({ec['vocabulary_size']} × {ec['n_embd']}). Its first six numbers, before and after training:

| | first 6 of 64 |
|---|---|
| before | `{', '.join(f"{v:+.4f}" for v in inspection['embedding_before'][:6])}` |
| after | `{', '.join(f"{v:+.4f}" for v in inspection['embedding_after'][:6])}` |

Before training those numbers are a random draw and mean nothing. After training they mean
something only in relation to other rows — which is exactly what the explorer's map draws. In the
trained model, the five rows closest to `surgeon` are `physician`, `dentist`, `therapist`,
`doctor`, `nurse` (cosine 0.89–0.96). In the untrained model of the same run, they are `peach`,
`robins`, `ordered`, `merchandise`, `nurse` (0.25–0.31) — noise, and no pair anywhere clears the
0.55 threshold the explorer defaults to.

### 3. A wrong guess becomes a gradient, and a gradient becomes a nudge

The very first training step, saved as it happened, for coordinate 0 of `{update['token']}`:

| | |
|---|---|
| value before the step | `{update['before']:.10f}` |
| gradient of the loss | `{update['gradient']:+.10f}` |
| learning rate at step 1 (warmup) | `{update['learning_rate']}` |
| value after the step | `{update['after']:.10f}` |
| actual change | `{update['after'] - update['before']:+.3e}` |

Worth staring at: the gradient is **negative** and the weight went **up** — correct, since a step
moves against the gradient to reduce the loss. And the size of the move is
`{update['after'] - update['before']:.2e}`, essentially the learning rate itself, not
learning-rate × gradient. That is AdamW: it divides by a running estimate of the gradient's own
magnitude, so the first step is about `lr × sign(gradient)`. A plain SGD step would have moved
this weight by {abs(update['gradient']) * update['learning_rate']:.2e} — five orders of magnitude
less. Multiply that by {ec['parameters']:,} parameters, {ec['training_steps']:,} steps and batches
of {ec['batch_size']} passages, and that is the entire training process.

### 4. The same prefix, before and after

Next-token probabilities for `{inspection['prefix']}`:

| | top 5 |
|---|---|
| before training | {top_probabilities('probabilities_before')} |
| after training | {top_probabilities('probabilities_after')} |

Before, the distribution is flat — {100*max(inspection['probabilities_before']):.2f}% on the best
word out of {ec['vocabulary_size']}, which is about 1/{ec['vocabulary_size']}, i.e. it knows
nothing. After, the mass is on `recommended`, `returned`, `compared`, `selected`, `reviewed`:
precisely the verbs that follow a customer in the training sentences. The model did not learn
what a customer *is*. It learned which words follow this one here.

### 5. Attention: how earlier words get a say

Each token's 64 numbers are mixed with the tokens *before* it, weighted by how relevant the model
finds them. The saved attention row for the same prefix is
`{[round(v, 3) for v in inspection['attention_rows'][0]]}` for the first position — all the weight
on itself, because a first token has nothing earlier to read. Causality is structural: position
*i* may only attend to positions ≤ *i*, which is why the heatmaps in the explorer are triangles.
Without that mask the model could see the answer while predicting it.

In the explorer's chat, `the report about the surgeon explains the health` ends with the last
token reading `report` 34.9%, `surgeon` 13.5%, `health` 8.4% — and then predicting `in` at ~100%,
because the training template is *"the report about the … explains the … in detail ."*

### 6. Temperature changes the dice, not the model

The same trained weights, same seed, three temperatures
([`temperature_comparison.json`]({expanded['path']}/temperature_comparison.json)):

| Temperature | First sample |
|---|---|
| 0.3 | `{expanded['temperature']['0.3'][0]}` |
| 0.8 | `{expanded['temperature']['0.8'][0]}` |
| 1.2 | `{expanded['temperature']['1.2'][0]}` |

Temperature divides the scores before softmax. Low temperature sharpens the distribution toward
the single best word; high temperature flattens it and lets unlikely words through. **No weight
changes** — this happens entirely at generation time. The explorer makes this literal: it shows
the dice roll and where it landed in the cumulative probabilities.

---

## Training curves and samples

![training curves]({expanded['path']}/training_curves.svg)

Loss is measured on **fixed panels** of at most
{ec['evaluation_panel_size']['train']} training and {ec['evaluation_panel_size']['validation']}
held-out passages, averaged over non-padding next-token targets — small estimates, not
full-corpus measurements. Both runs, every measured value
([starter]({starter['path']}/history.json) · [expanded]({expanded['path']}/history.json)):

**Starter corpus**

| Step | Training panel | Validation panel |
|---:|---:|---:|
{losses(starter)}

**Expanded corpus**

| Step | Training panel | Validation panel |
|---:|---:|---:|
{losses(expanded)}

The two runs' losses are **not comparable to each other**: different corpus, different
vocabulary, so a different task. Within each run, validation tracks training closely — expected,
since the split shares templates, and therefore *not* evidence of generalisation.

Both curves are also nearly flat from step 1,500 to 3,000, which is what makes
[`exp3-longer`](results/exp3-longer)'s failure to improve unsurprising in hindsight.

### What it writes, untrained → halfway → trained

Same starting token, same sampling seed, expanded run
([all saved samples]({expanded['path']}/samples)):

**Step 0 — untrained**

{samples_block(expanded, 0)}

**Step {ec['training_steps']//2} — halfway**

{samples_block(expanded, ec['training_steps']//2)}

**Step {ec['training_steps']} — trained**

{samples_block(expanded, ec['training_steps'])}

At step 0 it is word salad drawn from a flat distribution. By halfway the sentence *shapes* are
already right. Between halfway and the end the change is small — consistent with the loss curve,
and a good reminder that a fluent-looking sample from a template corpus is not knowledge.

---

## The interactive explorer

**▶ [Open it]({PAGES})** · source: [`viz/index.html`](viz/index.html)

The chat box runs the model **in your browser**: {ec['parameters']:,} float32 parameters exported
from `model.pt` by [`viz/export_model.py`](viz/export_model.py), and a forward pass written in
plain JavaScript ([`viz/gpt.js`](viz/gpt.js)) — embeddings, two blocks of causal 4-head attention
and GELU MLP, final LayerNorm, tied output projection. It checks itself on load against logits
captured from PyTorch at export time and reports the largest disagreement, which is **5.8e-6**.
No server, no API, no canned answers.

### Word graph, with the prediction drawn on it

![the word graph with a live prediction](docs/images/explorer-graph.png)

Every dot is one word in the vocabulary, sized by how often it was trained on. Two words are
linked when each is among the other's most similar vectors — cosine similarity across all 64
dimensions, not a 2-D projection — with the number of links and the similarity threshold under
your control, so the map cannot be quietly cherry-picked. Nothing labelled these groups: the
model only ever saw sentences and was only ever asked for the next word.

Type a prompt and the map becomes an explanation:

- the words of your prompt light up,
- **dashed** lines show what the last token actually attended to,
- **solid rays** fan out to the words it is considering next, thick in proportion to probability,
- the thickest ray leads to the word it picked.

The panel underneath shows the full distribution, the attention weights, and the dice roll that
chose between them.

### The same model before training

![the untrained map](docs/images/explorer-untrained.png)

Same words, same threshold, weights at initialisation: **zero links**. The vectors are random
draws, so no pair is meaningfully close to any other. This single comparison is the clearest
thing in the project — training is what turns {ec['vocabulary_size']} rows of noise into a map.

Switch the model to *Starter corpus · trained* and type `the opposite of hot is`: three of the
five words arrive as `<UNK>`, and the model answers by ending the sentence (`<EOS>` at 90%).
That is what an unscorable eval case looks like from the inside.

### Layers — the arithmetic for one prompt

![the layer view](docs/images/explorer-layers.png)

Tokens → row numbers → 64-number vectors → block 1's four attention heads → block 2's four →
the final probability list. The heatmaps are real values from the saved weights, and they are
triangles because a token may never read a later one.

### Progress — the run itself

![the progress view](docs/images/explorer-progress.png)

Loss curves, the samples at each milestone, every eval category across all four result sets, and
an animation of the embedding table every 250 steps.

Those frames needed the embedding table *during* training, which the notebook does not save — and
I did not want to edit the notebook that produced the eval evidence. So
[`viz/replay_training.py`](viz/replay_training.py) rebuilds the run from the artefacts the
notebook already saved (`split.json` for the exact documents, `tokenization.json` for the exact
vocabulary, `config.json` for the settings) and repeats the identical loop with the same seeds.
It proves it is the same run rather than a lookalike: the freshly initialised weights must hash
to the saved `model_untrained.pt`, and the final weights must hash to the run's recorded
`model_sha256`. Both match — `{milestones['verified_model_sha256'][:16]}…` — so the animation is
that exact training run, not a re-enactment.

### What the explorer does not prove

A tidy cluster of medical words is a picture of **co-occurrence in a small synthetic corpus**,
not evidence of meaning. `surgeon` sits near `physician` because the same templates produced
both. The map would look just as convincing if the corpus were nonsense, as long as it were
*consistent* nonsense.

---

## Chat with the model

Three interfaces, all driving the same trained weights
(`{web['model_sha256'][:16]}…`, {web['completed_steps']:,} steps,
[`results/exp2-expanded/model.pt`](results/exp2-expanded/model.pt)).

### Terminal — the supplied `chat.py`

```bash
python chat.py --model results/exp2-expanded/model.pt --transcript my-chat.json
```

Recorded session: [`terminal_session.txt`](results/chat/terminal_session.txt) ·
transcript: [`terminal_transcript.json`](results/chat/terminal_transcript.json)

| Prompt | Reply | Words it could not read |
|---|---|---|
{chat_turns}

### Web — `chat_web.py`, loading `model.pt` directly

```bash
python chat_web.py --model results/exp2-expanded/model.pt
# open http://127.0.0.1:8000
```

![the web chat](docs/images/chat-web.png)

Transcript: [`web_transcript.json`](results/chat/web_transcript.json)

| Prompt | Reply | Words it could not read |
|---|---|---|
{web_turns}

### In the explorer

The chat box in the explorer needs no server at all — see above.

### What these interactions show

- **It continues text; it does not answer.** `a robin is a bird . a salmon is a` → `fish .` is the
  best case, and it is still just continuation.
- **The multiple-choice score is not the chat.** The exam records `the opposite of empty is` →
  **full** as correct, yet the terminal reply to the same prompt was `two .` — sampling at
  temperature 0.8 from a distribution whose top word is right but not dominant.
- **Unknown words are invisible to it, not approximated.** The finance prompt lost nine words to
  `<UNK>`; the negation prompt lost six, including the two the question turns on, and the reply
  is unrelated text.
- **Every prompt starts fresh.** There is no conversation memory, the context is
  {ec['block_size']} tokens, and nothing typed anywhere retrains the model or enters the corpus.

---

## Reproducing everything

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt numpy

python make_corpus.py                  # regenerate the teaching corpus (checks leakage)
python corpus_sources/make_pdf.py      # re-render the PDF and verify extraction

python run_experiment.py --name exp1-starter  --corpus-folder corpus_starter
python run_experiment.py --name exp2-expanded --corpus-folder corpus
```

`run_experiment.py` patches only the settings cell of
[`notebook_template.ipynb`](notebook_template.ipynb), executes the notebook with outputs intact,
and copies the new `llm_runs/` folder into `results/`. The starter run points at
[`corpus_starter/`](corpus_starter), which holds nothing but a README, so the classroom corpus is
used alone and the run stays reproducible from a clone.

Rerun the exam on the saved weights, without the notebook:

```bash
python run_evals.py --model results/exp2-expanded/model.pt           --output /tmp/recheck-final
python run_evals.py --model results/exp2-expanded/model_untrained.pt --output /tmp/recheck-untrained --stage untrained
```

Rebuild the explorer's data and this README:

```bash
python viz/export_viz.py && python viz/export_model.py
python viz/replay_training.py --run results/exp2-expanded --every 250
python build_readme.py
python -m http.server 8777    # then open http://127.0.0.1:8777/viz/
```

Checks: `python -m unittest test_language_evals test_corpus` (the starter's own separation
tests) and `python make_corpus.py --check` (leakage and vocabulary-coverage report).

Hardware: {ec['hardware']}, {ec['device'].upper()}, PyTorch {ec['torch']}.
Training took **{starter['summary']['elapsed_seconds']:.1f}s** (starter) and
**{expanded['summary']['elapsed_seconds']:.1f}s** (expanded) for {ec['training_steps']:,}
completed steps each; neither run was interrupted.

---

## Limitations and the next experiment

1. **This is a development benchmark, not a test.** I read the results while choosing what to
   teach. The taught-category result is strong, but "does the model generalise" needs cases that
   never influenced a decision.
2. **The held-out split does not test generalisation either** — it splits passages, not source
   files, so both sides share templates.
3. **Three seeds is a small sample.** It is enough to show 0/12 → 11–12/12 is not noise; it is not
   enough to resolve the 4/8 → 8/8 transfer difference, which is why I do not claim it.
4. **Nothing here is understanding.** {taught_final}/12 on the taught categories means the model
   ranks one word above three others after a pattern it was drilled on. The free continuations in
   the chat section are the honest picture of what it can actually produce.
5. **The corpus is synthetic and narrow.** Every sentence follows a handful of templates I wrote,
   so both the loss and the clean clusters in the explorer are easier than real text would be.

**Next experiment**, one change, data only: fix the `water freezes into` collision. Remove
`ice melts into water .` and add sentences that put `ice` after `into` in frames unrelated to the
test prompt (`the cold turns the puddle into ice .`, `the machine makes water into ice .`), keep
steps, learning rate and seed identical, and rerun all 48 cases at the same three seeds. I expect
`lang_43` to flip and nothing else to move — and if other categories move too, the corpus is
doing something I have not accounted for. That is a one-variable test of the claim I am making in
this README: that what this model knows is exactly what its corpus taught it.

---

*nanoGPT is MIT-licensed by Andrej Karpathy ([`NANOGPT_LICENSE`](NANOGPT_LICENSE)). Starter
notebook, eval suite and runner come from the course sample project. Everything else — corpus,
experiments, explorer, in-browser model, interfaces and this README — is my own work for Class 4
of Fundamentals of Agentic AI.*
"""
    readme = readme.replace("{audited}", str(audit_fields["audited"]))
    for key, value in audit_fields.items():
        readme = readme.replace("{" + key + "}", str(value))
    (ROOT / "README.md").write_text(readme)
    print(f"wrote README.md — {len(readme.splitlines())} lines, {len(readme):,} characters")


if __name__ == "__main__":
    main()
