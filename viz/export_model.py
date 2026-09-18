"""Export the trained weights so the explorer can run the model in the browser.

The page is static — GitHub Pages cannot execute Python — so the chat box in the explorer
runs this nanoGPT itself, in JavaScript. That is only possible because the model is tiny:
122,368 parameters is 490 KB of float32. The same weights the notebook trained and the
evals scored are the ones that answer in the browser; nothing is precomputed or canned.

Writes, per state:
    viz/data/<state>.bin     every tensor, float32 little-endian, concatenated
    viz/data/<state>.json    vocabulary, config, and each tensor's offset and shape

The manifest also carries a handful of reference logits straight from PyTorch, which the
JavaScript implementation checks itself against on load.

    python viz/export_model.py
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_evals import load_model, model_hash, word_tokens  # noqa: E402

DATA = ROOT / "viz/data"
STATES = {"exp2-expanded:trained": ("exp2-expanded", "model.pt"),
          "exp2-expanded:untrained": ("exp2-expanded", "model_untrained.pt"),
          "exp1-starter:trained": ("exp1-starter", "model.pt"),
          "exp1-starter:untrained": ("exp1-starter", "model_untrained.pt")}
CHECK_PROMPTS = ["the report about the surgeon explains the", "the opposite of hot is",
                 "a robin is a bird . a salmon is a", "yesterday she", "water freezes into"]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (experiment, filename) in STATES.items():
        model, vocabulary, saved = load_model(ROOT / "results" / experiment / filename)
        tensors, offset, blob = {}, 0, bytearray()
        for key, tensor in model.state_dict().items():
            if key == "lm_head.weight":      # tied to transformer.wte.weight, stored once
                continue
            flat = tensor.detach().cpu().contiguous().float().flatten()
            blob += struct.pack(f"<{flat.numel()}f", *flat.tolist())
            tensors[key] = {"offset": offset, "shape": list(tensor.shape)}
            offset += flat.numel()

        stoi = {token: index for index, token in enumerate(vocabulary)}
        checks = []
        for prompt in CHECK_PROMPTS:
            ids = [stoi["<BOS>"]] + [stoi.get(t, stoi["<UNK>"]) for t in word_tokens(prompt)]
            with torch.no_grad():
                logits = model(torch.tensor([ids]))[0][0, -1]
            top = torch.argsort(logits, descending=True)[:5].tolist()
            checks.append({"prompt": prompt, "ids": ids,
                           "top_tokens": [vocabulary[i] for i in top],
                           "top_logits": [round(logits[i].item(), 5) for i in top]})

        slug = name.replace(":", "_")
        (DATA / f"{slug}.bin").write_bytes(bytes(blob))
        (DATA / f"{slug}.json").write_text(json.dumps({
            "state": name, "experiment": experiment, "weights_file": f"{slug}.bin",
            "model_sha256": model_hash(model), "vocabulary": vocabulary,
            "config": {"n_layer": model.config.n_layer, "n_head": model.config.n_head,
                       "n_embd": model.config.n_embd, "block_size": model.config.block_size,
                       "vocab_size": model.config.vocab_size,
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "completed_steps": saved.get("completed_steps")},
            "tensors": tensors, "float_count": offset, "checks": checks}, separators=(",", ":")) + "\n")
        print(f"{name}: {offset:,} floats ({offset*4/1e3:.0f} KB), "
              f"vocabulary {len(vocabulary)}, hash {model_hash(model)[:12]}…")


if __name__ == "__main__":
    main()
