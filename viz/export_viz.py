"""Export everything the interactive explorer shows, straight from the saved runs.

Nothing here is illustrative or hand-made: the embeddings come from the checkpoints the
notebook wrote, the attention numbers come from running the saved `model.pt` forward, and
the scores come from the eval summaries. Run after the experiments:

    python viz/export_viz.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_evals import load_model, load_suite, word_tokens  # noqa: E402

DATA = ROOT / "viz/data"
STARTER, EXPANDED = "exp1-starter", "exp2-expanded"
TAUGHT = ["grammar", "opposites", "everyday_knowledge", "categories_and_analogies"]
UNTAUGHT = ["negation", "reference", "sequence", "spatial_relations"]
STARTER_GROUPS = ["domain_context", "domain_place", "new_wording"]

# One prompt per thing worth looking at: a starter pattern, a transfer phrasing, each
# taught category, and a control case the model cannot even read.
LAYER_PROMPTS = [
    ("the report about the surgeon explains the", "starter pattern the corpus teaches directly"),
    ("yesterday our office discussed the platform and the", "starter words in a new sentence shape"),
    ("the opposite of hot is", "taught: opposites"),
    ("yesterday she", "taught: grammar, past tense after a time word"),
    ("a robin is a bird . a salmon is a", "taught: analogy across a sentence boundary"),
    ("water freezes into", "taught: everyday knowledge — and the one taught case that fails"),
    ("ava did not buy tea . she bought milk . ava bought", "control: never taught, mostly unknown words"),
]


def rounded(values, places=4):
    return [round(float(v), places) for v in values]


def eval_word_roles(suite):
    """Which eval case each vocabulary word belongs to, for colouring the graph."""
    roles = {}
    for case in suite["cases"]:
        for word in word_tokens(case["prompt"]):
            roles.setdefault(word, []).append({"case": case["id"], "category": case["category"],
                                               "group": case["group"], "role": "prompt"})
        for choice in case["choices"]:
            word = word_tokens(choice)[0]
            roles.setdefault(word, []).append({
                "case": case["id"], "category": case["category"], "group": case["group"],
                "role": "answer" if choice == case["answer"] else "distractor"})
    return roles


def graph_payload(suite):
    states, vocabularies = [], {}
    for experiment, label in [(STARTER, "Starter corpus"), (EXPANDED, "Expanded corpus")]:
        checkpoint = json.loads((ROOT / "results" / experiment / "checkpoint.json").read_text())
        vocabulary = checkpoint["vocabulary"]
        vocabularies[experiment] = vocabulary
        for stage, key in [("untrained", "initial_embeddings"), ("trained", None)]:
            table = checkpoint["initial_embeddings"] if key else checkpoint["weights"]["wte"]
            states.append({"id": f"{experiment}:{stage}", "experiment": experiment, "stage": stage,
                           "label": f"{label} · {stage}", "vocabulary": vocabulary,
                           "counts": checkpoint["token_counts"],
                           "embeddings": [rounded(row) for row in table]})
    starter_types = set(vocabularies[STARTER])
    return {"states": states,
            "new_in_expanded": sorted(set(vocabularies[EXPANDED]) - starter_types),
            "eval_words": eval_word_roles(suite),
            "taught": TAUGHT, "untaught": UNTAUGHT,
            "note": ("Vectors are the token embedding rows saved by each run: "
                     "checkpoint.json's initial_embeddings for untrained, weights.wte for trained.")}


@torch.no_grad()
def trace(model, vocabulary, prompt):
    """Real activations: attention per block and head, plus the next-word distribution."""
    stoi = {token: index for index, token in enumerate(vocabulary)}
    tokens = word_tokens(prompt)
    ids = [stoi["<BOS>"]] + [stoi.get(t, stoi["<UNK>"]) for t in tokens]
    ids = ids[-model.config.block_size:]
    index = torch.tensor([ids])
    n_embd, n_head = model.config.n_embd, model.config.n_head
    head_size, length = n_embd // n_head, len(ids)
    x = model.transformer.wte(index) + model.transformer.wpe(torch.arange(length))
    embeddings = [rounded(model.transformer.wte.weight[i].tolist(), 3) for i in ids]
    blocks = []
    for block in model.transformer.h:
        normed = block.ln_1(x)
        q, k, _ = block.attn.c_attn(normed).split(n_embd, dim=-1)
        q = q.view(1, length, n_head, head_size).transpose(1, 2)
        k = k.view(1, length, n_head, head_size).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)
        scores = scores.masked_fill(torch.triu(torch.ones(length, length), diagonal=1).bool(), float("-inf"))
        attention = F.softmax(scores, dim=-1)[0]
        blocks.append([[rounded(row, 4) for row in head.tolist()] for head in attention])
        x = block(x)
    logits = model(index)[0][0, -1]
    probabilities = F.softmax(logits, dim=-1)
    order = torch.argsort(probabilities, descending=True)[:10].tolist()
    return {"tokens": ["<BOS>"] + [vocabulary[i] if i < len(vocabulary) else "?" for i in ids[1:]],
            "ids": ids, "unknown": sorted({t for t in tokens if t not in stoi}),
            "embeddings": embeddings, "attention": blocks,
            "top": [{"token": vocabulary[i], "probability": round(probabilities[i].item(), 5)} for i in order]}


def layers_payload():
    models = {}
    for experiment in (STARTER, EXPANDED):
        for stage, filename in [("untrained", "model_untrained.pt"), ("trained", "model.pt")]:
            model, vocabulary, _ = load_model(ROOT / "results" / experiment / filename)
            models[f"{experiment}:{stage}"] = (model, vocabulary)
    prompts = []
    for prompt, why in LAYER_PROMPTS:
        entry = {"prompt": prompt, "why": why, "states": {}}
        for state, (model, vocabulary) in models.items():
            entry["states"][state] = trace(model, vocabulary, prompt)
        prompts.append(entry)
    model, _ = models[f"{EXPANDED}:trained"]
    return {"prompts": prompts,
            "architecture": {"blocks": model.config.n_layer, "heads": model.config.n_head,
                             "n_embd": model.config.n_embd, "block_size": model.config.block_size,
                             "parameters": sum(p.numel() for p in model.parameters())}}


def run_summary(name):
    base = ROOT / "results" / name
    if not base.exists():
        return None
    samples = {path.stem.replace("step_", "").lstrip("0") or "0": path.read_text().rstrip("\n").split("\n")
               for path in sorted((base / "samples").glob("*.txt"))}
    stages = {stage: json.loads((base / "language_evals" / stage / "eval_summary.json").read_text())
              for stage in ("untrained", "final")}
    cases = {stage: json.loads((base / "language_evals" / stage / "eval_results.json").read_text())
             for stage in ("untrained", "final")}
    config = json.loads((base / "config.json").read_text())
    return {"name": name, "config": {k: config[k] for k in (
                "training_steps", "learning_rate", "seed", "vocabulary_size", "parameters",
                "train_documents", "validation_documents", "corpus_files", "corpus_mode",
                "training_unknown_rate", "validation_unknown_rate", "evaluation_panel_size")},
            "training_summary": json.loads((base / "training_summary.json").read_text()),
            "history": json.loads((base / "history.json").read_text()),
            "samples": samples, "eval": stages,
            "temperature": json.loads((base / "temperature_comparison.json").read_text()),
            "inspection": {k: v for k, v in json.loads((base / "inspection.json").read_text()).items()
                           if k != "attention_rows"},
            "cases": {stage: [{"id": row["id"], "category": row["category"], "group": row["group"],
                               "prompt": row["prompt"], "expected": row["expected"],
                               "predicted": row["predicted_choice"], "score": row["score"],
                               "status": row["status"], "generated": row["generated_text"],
                               "probabilities": {k: round(v, 5) for k, v in row["choice_probabilities"].items()}}
                              for row in rows] for stage, rows in cases.items()}}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    suite = load_suite(ROOT / "evals/language_evals.json")

    for name, payload in [("graph", graph_payload(suite)), ("layers", layers_payload())]:
        path = DATA / f"{name}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        print(f"wrote {path.relative_to(ROOT)} — {path.stat().st_size/1e6:.2f} MB")

    runs = [name for name in (STARTER, EXPANDED, "exp3-longer", "rb-starter-s1234",
                              "rb-expanded-s1234", "rb-starter-s2468", "rb-expanded-s2468")]
    progress = {"runs": {name: run_summary(name) for name in runs if run_summary(name)},
                "taught": TAUGHT, "untaught": UNTAUGHT, "starter_groups": STARTER_GROUPS,
                "suite_sha256": json.loads((ROOT / "results" / EXPANDED /
                                            "language_evals/final/eval_summary.json").read_text())["suite_sha256"]}
    path = DATA / "progress.json"
    path.write_text(json.dumps(progress, separators=(",", ":")) + "\n")
    print(f"wrote {path.relative_to(ROOT)} — {path.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
