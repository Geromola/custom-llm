"""Re-run one saved experiment step for step to capture the embedding table as it moves.

The notebook only saves the embedding table twice: at initialisation and at the end. The
progress animation needs the in-between frames, and I did not want to edit the notebook
that produced the eval evidence. So this script rebuilds the same run from the artefacts
the notebook already saved (`split.json` for the exact documents, `tokenization.json` for
the exact vocabulary, `config.json` for the settings) and repeats the identical training
loop with the same seeds.

It proves the replay is the same run rather than a lookalike: the weight hash of the
freshly initialised model must equal the notebook's `model_untrained.pt` hash, and the
weight hash after the last step must equal the hash recorded in the run's final eval
summary. If either check fails the script stops instead of writing misleading frames.

    python viz/replay_training.py --run results/exp2-expanded --every 250
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_evals import model_hash  # noqa: E402


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--every", type=int, default=250)
    parser.add_argument("--output", type=Path, default=ROOT / "viz/data/milestones.json")
    args = parser.parse_args()

    run = (ROOT / args.run) if not args.run.is_absolute() else args.run
    config = json.loads((run / "config.json").read_text())
    split = json.loads((run / "split.json").read_text())
    vocabulary = json.loads((run / "tokenization.json").read_text())["vocabulary"]
    train_docs = split["train"]

    import importlib.util
    spec = importlib.util.spec_from_file_location("classroom_nanogpt", ROOT / "nanogpt_model.py")
    nanogpt = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = nanogpt
    spec.loader.exec_module(nanogpt)

    seed = config["seed"]
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)          # exactly as the notebook does before building the model
    model_config = nanogpt.GPTConfig(vocab_size=len(vocabulary), block_size=config["block_size"],
                                     n_layer=config["n_layer"], n_head=config["n_head"],
                                     n_embd=config["n_embd"], dropout=0.0, bias=True)
    model = nanogpt.GPT(model_config)

    saved_untrained = torch.load(run / "model_untrained.pt", map_location="cpu", weights_only=True)
    if model_hash(model) != state_hash(saved_untrained["model"]):
        raise SystemExit("Replay initialisation differs from the saved untrained model.")

    stoi = {token: index for index, token in enumerate(vocabulary)}
    UNK, BOS, EOS = 0, 1, 2
    import re

    def word_tokens(text):
        return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text.lower(), flags=re.UNICODE)

    def tokenize(doc):
        return [BOS] + [stoi.get(t, UNK) for t in word_tokens(doc)] + [EOS]

    def batch(documents):
        sequences = [tokenize(doc) for doc in documents]
        length = max(len(seq) - 1 for seq in sequences)
        x = torch.full((len(sequences), length), EOS, dtype=torch.long)
        y = torch.full_like(x, -1)
        for i, seq in enumerate(sequences):
            x[i, :len(seq) - 1] = torch.tensor(seq[:-1])
            y[i, :len(seq) - 1] = torch.tensor(seq[1:])
        return x, y

    steps, learning_rate, batch_size = config["training_steps"], config["learning_rate"], config["batch_size"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(.9, .95), weight_decay=.01)
    sampling_rng = random.Random(seed + 1)
    frames = [{"step": 0, "embeddings": model.transformer.wte.weight.detach().cpu().tolist()}]
    for step in range(steps):
        warmup = min(100, max(1, steps // 10))
        progress = max(0, step - warmup) / max(1, steps - warmup)
        lr = learning_rate * min(1, (step + 1) / warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(*batch(sampling_rng.choices(train_docs, k=batch_size)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        completed = step + 1
        if completed % args.every == 0 or completed == steps:
            frames.append({"step": completed,
                           "embeddings": model.transformer.wte.weight.detach().cpu().tolist()})
            print(f"  step {completed:5d}  batch loss {loss.item():.4f}", flush=True)

    expected = json.loads((run / "language_evals/final/eval_summary.json").read_text())["model_sha256"]
    actual = model_hash(model)
    if actual != expected:
        raise SystemExit(f"Replay diverged: {actual[:16]} != notebook {expected[:16]}")
    print("replay weight hash matches the notebook run:", actual[:16], "...")

    payload = {"run": str(args.run), "steps": steps, "every": args.every,
               "vocabulary": vocabulary, "verified_model_sha256": actual,
               "verification": "final replay weights are bit-identical to the notebook run",
               "frames": [{"step": f["step"],
                           "embeddings": [[round(v, 4) for v in row] for row in f["embeddings"]]}
                          for f in frames]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    size = args.output.stat().st_size / 1e6
    print(f"wrote {args.output.relative_to(ROOT)} — {len(frames)} frames, {size:.1f} MB")


if __name__ == "__main__":
    main()
