"""
Smoke test: run predict.py end-to-end on the bundled T. gondii example
and verify the output schema and a few sanity invariants.

This test downloads ESM-2 (~2.5 GB) the first time it runs, so it is slow
on a fresh CI runner. It is the only test, on purpose: it exercises the
full pipeline (parse, embed, classify, structural lookup, write).
"""
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parent.parent
EXPECTED_COLUMNS = [
    "protein_id",
    "description",
    "length",
    "predicted_crispr_score",
    "score_std",
    "essential_probability",
    "essentiality_class",
    "essentiality_confidence",
    "predicted_compartment",
    "compartment_confidence",
    "invasion_probability",
    "contrastive_score",
    "predicted_invasion",
    "similar_1_id",
    "similar_1_desc",
    "similar_1_compartment",
    "similar_1_similarity",
    "similar_1_crispr",
    "similar_2_id",
    "similar_2_desc",
    "similar_2_compartment",
    "similar_2_similarity",
    "similar_2_crispr",
    "similar_3_id",
    "similar_3_desc",
    "similar_3_compartment",
    "similar_3_similarity",
    "similar_3_crispr",
    "max_similarity_to_known",
    "structural_novelty",
    "match_specificity",
    "invasion_specific",
]


def test_smoke_tg(tmp_path):
    out = tmp_path / "predictions.tsv"
    result = subprocess.run(
        [sys.executable, str(REPO / "predict.py"),
         "--input", str(REPO / "examples/test_tg.fasta"),
         "--output", str(out),
         "--device", "cpu",
         "--batch-size", "2"],
        capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, (
        f"predict.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert out.exists(), "predict.py did not write the output file"

    df = pd.read_csv(out, sep="\t")

    # All 10 example proteins should be present
    assert len(df) == 10, f"expected 10 predictions, got {len(df)}"

    # Schema must match what the README documents
    assert list(df.columns) == EXPECTED_COLUMNS, (
        "output schema drifted: "
        f"missing {set(EXPECTED_COLUMNS) - set(df.columns)}, "
        f"extra {set(df.columns) - set(EXPECTED_COLUMNS)}"
    )

    # Sanity: characterised proteins should self-match in the reference DB at
    # similarity ~1.0. Proteins with "unknown" compartment in the example file
    # are excluded from reference_db.npz, so we allow up to one near-miss.
    n_self_match = (df["max_similarity_to_known"] >= 0.99).sum()
    assert n_self_match >= 9, (
        f"expected >=9 of 10 example proteins to self-match in the reference "
        f"DB at sim>=0.99, got {n_self_match}"
    )

    # Sanity: invasion compartments should dominate the example set
    # (8/10 are invasion compartments by construction).
    n_invasion = (df["predicted_invasion"] == "yes").sum()
    assert n_invasion >= 7, f"expected >=7 invasion calls, got {n_invasion}"

    # Sanity: essential probabilities are valid probabilities
    assert df["essential_probability"].between(0, 1).all()
    assert df["invasion_probability"].between(0, 1).all()
    assert df["compartment_confidence"].between(0, 1).all()
