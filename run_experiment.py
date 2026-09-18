"""Execute the notebook once with a given corpus folder and archive the run.

Every experiment in this repository was produced with this script, so the four eval
result sets differ only in the settings printed in each notebook's first cell.

    python run_experiment.py --name exp1-starter --corpus-folder corpus_starter
    python run_experiment.py --name exp2-expanded --corpus-folder corpus
    python run_experiment.py --name exp3-longer --corpus-folder corpus --steps 9000

The executed notebook keeps all outputs. The new `llm_runs/<timestamp>/` folder is copied
to `results/<name>/` and the ZIP is kept next to it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "notebook_template.ipynb"


def configure(settings: dict) -> dict:
    notebook = json.loads(TEMPLATE.read_text())
    cell = notebook["cells"][1]
    source = "".join(cell["source"])
    assert source.startswith("CORPUS ="), "the settings cell moved; check the template"
    cell["source"] = [
        f'CORPUS = "{settings["corpus"]}"       # Teaching sentences + files; "folder" uses only files\n',
        f'CORPUS_FOLDER = "{settings["corpus_folder"]}"   # Add .pdf, .txt and .md files here, including subfolders\n',
        f'TRAINING_STEPS = {settings["steps"]}      # 10 for setup; 3000 for the main experiment\n',
        f'LEARNING_RATE = {settings["learning_rate"]}\n',
    ]
    if settings["seed"] != 42:
        # The only way to vary the seed: it is a constant in the setup cell.
        for other in notebook["cells"]:
            source = "".join(other["source"])
            if "SEED, N_EMBD, N_HEAD" in source:
                other["source"] = [line.replace(
                    "SEED, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, BATCH_SIZE = 42,",
                    f"SEED, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, BATCH_SIZE = {settings['seed']},")
                    for line in other["source"]]
                break
        else:
            raise SystemExit("could not find the seed constant")
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="results/<name> and the notebook suffix")
    parser.add_argument("--corpus-folder", required=True)
    parser.add_argument("--corpus", default="classroom")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42, help="only for the robustness runs")
    parser.add_argument("--notebook", help="where to write the executed notebook")
    args = parser.parse_args()

    from nbclient import NotebookClient
    import nbformat

    settings = {"corpus": args.corpus, "corpus_folder": args.corpus_folder,
                "steps": args.steps, "learning_rate": args.learning_rate, "seed": args.seed}
    notebook = nbformat.reads(json.dumps(configure(settings)), as_version=4)
    before = {p.name for p in (ROOT / "llm_runs").glob("*")} if (ROOT / "llm_runs").exists() else set()

    print(f"[{args.name}] executing with", settings, flush=True)
    started = time.perf_counter()
    NotebookClient(notebook, timeout=3600, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    print(f"[{args.name}] notebook finished in {time.perf_counter()-started:.1f}s", flush=True)

    target = Path(args.notebook) if args.notebook else ROOT / f"custom_llm_{args.name}.ipynb"
    nbformat.write(notebook, target)

    after = {p.name for p in (ROOT / "llm_runs").glob("*")}
    new = sorted(after - before - {p for p in after if p.endswith(".zip")})
    new = [n for n in new if (ROOT / "llm_runs" / n).is_dir()]
    if len(new) != 1:
        raise SystemExit(f"expected exactly one new run folder, found {new}")
    run_dir = ROOT / "llm_runs" / new[0]
    destination = ROOT / "results" / args.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(run_dir, destination)
    zip_source = run_dir.with_suffix(".zip")
    if zip_source.exists():
        shutil.copy2(zip_source, destination.with_suffix(".zip"))
    (destination / "run_identity.json").write_text(json.dumps(
        {"experiment": args.name, "llm_run": run_dir.name, "settings": settings,
         "executed_notebook": target.name}, indent=2) + "\n")
    print(f"[{args.name}] notebook -> {target.name} | results -> results/{args.name}")


if __name__ == "__main__":
    main()
