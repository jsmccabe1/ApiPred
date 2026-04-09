#!/usr/bin/env python3
"""
Train ApiPred models from T. gondii data.

Generates three files in models/:
  - essentiality_ensemble.joblib  (5-fold ensemble for scores + confidence)
  - compartment_model.joblib      (multi-class compartment classifier)
  - reference_db.npz               (embedding database for structural context)

Usage:
    python train_model.py --data-dir ~/Apicomplexa/

The --data-dir should contain:
  - results/embeddings/all_proteins/protein_embeddings.npy
  - results/embeddings/all_proteins/protein_ids.txt
  - data/processed/protein_features.tsv
  - data/processed/protein_compartments.tsv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import joblib
import warnings
warnings.filterwarnings("ignore")

INVASION_COMPARTMENTS = {
    "rhoptries 1", "rhoptries 2", "micronemes", "dense granules",
    "IMC", "apical 1", "apical 2",
}

# Minimum compartment size for multi-class prediction
MIN_COMPARTMENT_SIZE = 10


def main():
    parser = argparse.ArgumentParser(description="Train ApiPred models")
    parser.add_argument("--data-dir", required=True,
                        help="Path to Apicomplexa project directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for models")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds for ensemble (default: 5)")
    parser.add_argument("--fast", action="store_true",
                        help="Use fewer estimators for faster training")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = (Path(args.output_dir) if args.output_dir
                  else Path(__file__).resolve().parent / "models")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_est = 100 if args.fast else 300
    max_d = 3 if args.fast else 4
    lr = 0.05

    # ── Load data ──
    print("Loading T. gondii data...")
    emb = np.load(data_dir / "results/embeddings/all_proteins/protein_embeddings.npy")
    with open(data_dir / "results/embeddings/all_proteins/protein_ids.txt") as f:
        ids = [l.strip() for l in f]
    idx_map = {pid: i for i, pid in enumerate(ids)}

    feat_df = pd.read_csv(data_dir / "data/processed/protein_features.tsv", sep="\t")
    comp_df = pd.read_csv(data_dir / "data/processed/protein_compartments.tsv", sep="\t")
    comp_dict = dict(zip(comp_df["accession"], comp_df["compartment"]))
    desc_dict = dict(zip(comp_df["accession"],
                         comp_df.get("description", pd.Series(dtype=str))))
    crispr_dict = dict(zip(feat_df["Accession"], feat_df["CRISPR.Score"]))

    print(f"  {len(ids)} proteins, {len(feat_df)} with features, "
          f"{len(comp_df)} with compartments")

    # ══════════════════════════════════════════════════════════════
    # ESSENTIALITY ENSEMBLE (5-fold, each fold = independent model)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Training essentiality ensemble ({args.n_folds} folds)")
    print(f"{'='*60}")

    ess_rows = []
    for _, r in feat_df.iterrows():
        acc = r["Accession"]
        score = r["CRISPR.Score"]
        if acc in idx_map and pd.notna(score):
            ess_rows.append({"accession": acc, "crispr_score": score,
                             "idx": idx_map[acc]})
    ess_df = pd.DataFrame(ess_rows)

    X_ess = emb[ess_df["idx"].values]
    y_cont = ess_df["crispr_score"].values
    y_bin = (y_cont < -3).astype(int)

    print(f"  Training set: {len(ess_df)} proteins ({y_bin.sum()} essential)")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    ensemble = []
    oof_preds = np.zeros(len(y_cont))
    oof_probs = np.zeros(len(y_cont))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_ess)):
        X_tr, X_val = X_ess[train_idx], X_ess[val_idx]
        y_tr_c, y_val_c = y_cont[train_idx], y_cont[val_idx]
        y_tr_b, y_val_b = y_bin[train_idx], y_bin[val_idx]

        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(
                n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                subsample=0.8, random_state=42 + fold
            ))
        ])
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                subsample=0.8, random_state=42 + fold
            ))
        ])

        reg.fit(X_tr, y_tr_c)
        clf.fit(X_tr, y_tr_b)

        oof_preds[val_idx] = reg.predict(X_val)
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

        ensemble.append({"regressor": reg, "classifier": clf})

        val_rho = spearmanr(y_val_c, reg.predict(X_val))[0]
        val_auc = roc_auc_score(y_val_b, clf.predict_proba(X_val)[:, 1])
        print(f"  Fold {fold+1}: rho={val_rho:.3f}, AUC={val_auc:.3f}")

    overall_rho = spearmanr(y_cont, oof_preds)[0]
    overall_auc = roc_auc_score(y_bin, oof_probs)
    print(f"\n  Overall OOF: rho={overall_rho:.4f}, AUC={overall_auc:.4f}")

    joblib.dump({"ensemble": ensemble, "oof_rho": overall_rho,
                 "oof_auc": overall_auc},
                output_dir / "essentiality_ensemble.joblib")
    print(f"  Saved: essentiality_ensemble.joblib")

    # ══════════════════════════════════════════════════════════════
    # MULTI-CLASS COMPARTMENT MODEL
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Training multi-class compartment model")
    print(f"{'='*60}")

    # Build compartment training set (exclude unknowns, small compartments)
    comp_rows = []
    comp_counts = {}
    for pid in ids:
        comp = comp_dict.get(pid, "unknown")
        if comp != "unknown" and pid in idx_map:
            comp_counts[comp] = comp_counts.get(comp, 0) + 1

    # Filter to compartments with enough examples
    valid_comps = {c for c, n in comp_counts.items() if n >= MIN_COMPARTMENT_SIZE}
    print(f"  Compartments with >={MIN_COMPARTMENT_SIZE} proteins: "
          f"{len(valid_comps)}")

    for pid in ids:
        comp = comp_dict.get(pid, "unknown")
        if comp in valid_comps and pid in idx_map:
            comp_rows.append({"accession": pid, "compartment": comp,
                              "idx": idx_map[pid]})

    comp_train_df = pd.DataFrame(comp_rows)
    X_comp = emb[comp_train_df["idx"].values]
    y_comp = comp_train_df["compartment"].values
    classes = np.sort(np.unique(y_comp))

    print(f"  Training set: {len(comp_train_df)} proteins, "
          f"{len(classes)} compartments")
    for c in classes:
        n = (y_comp == c).sum()
        inv_flag = " *" if c in INVASION_COMPARTMENTS else ""
        print(f"    {c}: {n}{inv_flag}")

    # RandomForest for multi-class: natively handles many classes without
    # one-vs-rest decomposition, so ~25x faster than GBM for 25 classes
    comp_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", n_jobs=-1, random_state=42
        ))
    ])

    # Evaluate with stratified CV
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof_comp_probs = cross_val_predict(comp_clf, X_comp, y_comp,
                                        cv=skf, method="predict_proba")

    # Per-compartment AUC (one-vs-rest)
    print(f"\n  Per-compartment OVR AUC:")
    for i, c in enumerate(comp_clf.classes_ if hasattr(comp_clf, 'classes_')
                          else classes):
        y_ovr = (y_comp == c).astype(int)
        if y_ovr.sum() >= 5:
            try:
                auc = roc_auc_score(y_ovr, oof_comp_probs[:, i])
                inv_flag = " *invasion*" if c in INVASION_COMPARTMENTS else ""
                print(f"    {c}: AUC={auc:.3f} (n={y_ovr.sum()}){inv_flag}")
            except (ValueError, IndexError):
                pass

    # Overall accuracy
    oof_pred_classes = classes[np.argmax(oof_comp_probs, axis=1)]
    accuracy = (oof_pred_classes == y_comp).mean()
    print(f"\n  Overall accuracy: {accuracy:.3f}")

    # Invasion binary metrics from multi-class
    y_inv_true = np.array([c in INVASION_COMPARTMENTS for c in y_comp]).astype(int)
    inv_col_mask = np.array([c in INVASION_COMPARTMENTS for c in classes])
    oof_inv_probs = oof_comp_probs[:, inv_col_mask].sum(axis=1)
    inv_auc = roc_auc_score(y_inv_true, oof_inv_probs)
    print(f"  Invasion AUC (from multi-class): {inv_auc:.3f}")

    # Fit final model on all data
    comp_clf.fit(X_comp, y_comp)

    joblib.dump({"model": comp_clf, "classes": classes,
                 "accuracy": accuracy, "invasion_auc": inv_auc},
                output_dir / "compartment_model.joblib")
    print(f"  Saved: compartment_model.joblib")

    # ══════════════════════════════════════════════════════════════
    # REFERENCE DATABASE
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Building reference database")
    print(f"{'='*60}")

    ref_ids = []
    ref_descs = []
    ref_comps = []
    ref_scores = []
    ref_indices = []

    for pid in ids:
        comp = comp_dict.get(pid)
        if comp and comp != "unknown" and pid in idx_map:
            ref_ids.append(pid)
            ref_descs.append(str(desc_dict.get(pid, ""))[:80])
            ref_comps.append(comp)
            score = crispr_dict.get(pid, np.nan)
            ref_scores.append(float(score) if pd.notna(score) else np.nan)
            ref_indices.append(idx_map[pid])

    ref_embs = emb[ref_indices]
    print(f"  Reference database: {len(ref_ids)} characterised proteins")

    np.savez(output_dir / "reference_db.npz",
             embeddings=ref_embs,
             ids=np.array(ref_ids, dtype=object),
             descriptions=np.array(ref_descs, dtype=object),
             compartments=np.array(ref_comps, dtype=object),
             crispr_scores=np.array(ref_scores, dtype=float))
    print(f"  Saved: reference_db.npz")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"ApiPred model training complete")
    print(f"{'='*60}")
    print(f"  Essentiality ensemble: rho={overall_rho:.3f}, AUC={overall_auc:.3f}")
    print(f"  Compartment model: {len(classes)} classes, "
          f"accuracy={accuracy:.3f}, invasion AUC={inv_auc:.3f}")
    print(f"  Reference DB: {len(ref_ids)} proteins")
    print(f"  Models saved to: {output_dir}/")
    print(f"\n  To use: python predict.py --input proteome.fasta --output predictions.tsv")


if __name__ == "__main__":
    main()
