#!/usr/bin/env python3
"""
Build a baseline control proteome panel for ApiPred.

Runs predict.py on a set of reference proteomes (apicomplexan positive
controls + free-living alveolate and non-alveolate negative controls),
collects per-organism invasion_probability distributions, and writes a
compact panel.json that predict.py can load to compute apicomplexan_rank,
background_rank, and invasion_fdr columns for any new query.

Usage:
    python tools/build_panel.py --config tools/panel_default.json \\
        --output panel/ --device cuda --batch-size 8

The config JSON is a list of entries:
    [
        {"name": "Toxoplasma_gondii",
         "path": "/path/to/Toxoplasma_gondii.fasta",
         "category": "apicomplexan"},
        {"name": "Chromera_velia",
         "path": "/path/to/Chromera_velia_CCMP2878.fasta",
         "category": "background"},
        ...
    ]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_predict(predict_py, input_fasta, output_tsv, device, batch_size,
                model_dir=None):
    """Invoke predict.py on one proteome. Skips if output already exists."""
    if output_tsv.exists() and output_tsv.stat().st_size > 1000:
        print(f"  [cached] {output_tsv.name}")
        return
    cmd = [sys.executable, str(predict_py),
           "--input", str(input_fasta),
           "--output", str(output_tsv),
           "--device", device,
           "--batch-size", str(batch_size)]
    if model_dir:
        cmd += ["--model-dir", str(model_dir)]
    print(f"  Running predict.py on {input_fasta.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: predict.py failed for {input_fasta.name}")
        print(f"  STDOUT: {result.stdout[-500:]}")
        print(f"  STDERR: {result.stderr[-500:]}")
        sys.exit(1)


def build_fdr_table(background_scores, score_grid=None):
    """Compute empirical FDR as a function of invasion_probability threshold.

    FDR(threshold) = fraction of background proteins with score >= threshold.

    Returns a list of [threshold, fdr] pairs sorted by threshold ascending.
    Also returns an equivalent dense grid for predict.py's searchsorted lookup.
    """
    bg = np.sort(np.asarray(background_scores, dtype=np.float64))
    n = len(bg)
    if score_grid is None:
        # Use a fixed grid of 0-1 in 0.005 increments (201 points)
        score_grid = np.linspace(0, 1, 201)
    # FDR at threshold t = P(background >= t) = (n - searchsorted_left(bg, t)) / n
    left = np.searchsorted(bg, score_grid, side="left")
    fdr = (n - left) / float(n)
    return [[float(t), float(f)] for t, f in zip(score_grid, fdr)]


def main():
    parser = argparse.ArgumentParser(description="Build an ApiPred baseline panel")
    parser.add_argument("--config", required=True,
                        help="JSON list of organisms (name, path, category)")
    parser.add_argument("--output", required=True,
                        help="Output panel directory")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--predict-py", default=None,
                        help="Path to predict.py (default: sibling of this script)")
    args = parser.parse_args()

    predict_py = Path(args.predict_py) if args.predict_py else (
        Path(__file__).resolve().parent.parent / "predict.py")
    if not predict_py.exists():
        print(f"ERROR: predict.py not found at {predict_py}")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    organisms_dir = out_dir / "organisms"
    organisms_dir.mkdir(exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    apico_all = []
    bg_all = []
    organism_stats = []

    print(f"Building panel with {len(config)} organisms")
    print(f"Output: {out_dir}")
    print()

    for entry in config:
        name = entry["name"]
        path = Path(entry["path"])
        category = entry["category"]
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        print(f"[{category}] {name}")
        output_tsv = organisms_dir / f"{name}.tsv"
        run_predict(predict_py, path, output_tsv, args.device, args.batch_size,
                    model_dir=args.model_dir)

        # Load the per-organism predictions
        df = pd.read_csv(output_tsv, sep="\t", low_memory=False,
                         usecols=["invasion_probability", "predicted_compartment"])
        inv = df["invasion_probability"].dropna().astype(float).values
        organism_stats.append({
            "name": name,
            "category": category,
            "n_proteins": int(len(df)),
            "n_invasion_calls": int((inv > 0.5).sum()),
            "median_inv": float(np.median(inv)),
            "q99_inv": float(np.quantile(inv, 0.99)),
        })
        print(f"  {len(df)} proteins, {(inv > 0.5).sum()} with inv_prob>0.5 "
              f"(q99={np.quantile(inv, 0.99):.3f})")
        if category == "apicomplexan":
            apico_all.extend(inv.tolist())
        elif category == "background":
            bg_all.extend(inv.tolist())

    if not apico_all:
        print("ERROR: no apicomplexan organisms loaded")
        sys.exit(1)
    if not bg_all:
        print("ERROR: no background organisms loaded")
        sys.exit(1)

    apico_sorted = np.sort(np.asarray(apico_all, dtype=np.float64))
    bg_sorted = np.sort(np.asarray(bg_all, dtype=np.float64))

    print()
    print(f"Apicomplexan panel: {len(apico_sorted)} proteins "
          f"(median={np.median(apico_sorted):.3f}, q99={np.quantile(apico_sorted, 0.99):.3f})")
    print(f"Background panel:  {len(bg_sorted)} proteins "
          f"(median={np.median(bg_sorted):.3f}, q99={np.quantile(bg_sorted, 0.99):.3f})")

    # Build the FDR table from background
    fdr_table = build_fdr_table(bg_sorted)

    # Write the compact panel.json
    panel = {
        "version": "1.0",
        "organisms": organism_stats,
        "apicomplexan_invasion_probs": [round(float(x), 4) for x in apico_sorted],
        "background_invasion_probs": [round(float(x), 4) for x in bg_sorted],
        "background_fdr_table": fdr_table,
    }
    panel_path = out_dir / "panel.json"
    with open(panel_path, "w") as f:
        json.dump(panel, f)
    size_mb = panel_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {panel_path} ({size_mb:.1f} MB)")

    # Human-readable summary
    summary_path = out_dir / "panel_summary.tsv"
    pd.DataFrame(organism_stats).to_csv(summary_path, sep="\t", index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
