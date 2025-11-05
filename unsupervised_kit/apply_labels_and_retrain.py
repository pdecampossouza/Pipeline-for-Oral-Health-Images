# -*- coding: utf-8 -*-
"""
Retrain the baseline view classifier using labels from metadata/views.csv,
merging with manifest.csv to get pdf_base/voluntario/seq.

Outputs:
- console classification report
- metadata/view_confusion_matrix.png
- metadata/pred_view_results.csv  (filepath, y_true, y_pred, margin)

Usage:
  python unsupervised_kit/apply_labels_and_retrain.py --root "."
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix


def norm_paths(df: pd.DataFrame, col="filepath") -> pd.DataFrame:
    df[col] = df[col].astype(str).str.replace("\\\\", "\\").str.replace("/", "\\")
    return df


def load_artifacts(root: Path):
    meta = root / "metadata"
    X = np.load(meta / "features.npy")  # (N, D) aligned with manifest rows
    man = pd.read_csv(meta / "manifest.csv")
    views = pd.read_csv(meta / "views.csv")
    man = norm_paths(man, "filepath")
    views = norm_paths(views, "filepath")
    return X, man, views


def split_by_volunteer(df_labeled: pd.DataFrame):
    """70/30 split by volunteer, per pdf_base (avoids leakage)."""
    train_idx, test_idx = [], []
    for pdf, g in df_labeled.groupby("pdf_base"):
        vols = sorted(g["voluntario"].astype(int).unique())
        n = len(vols)
        n_train = max(1, int(0.7 * n))
        train_vols = set(vols[:n_train])
        test_vols = set(vols[n_train:])
        for i, r in g.iterrows():
            (train_idx if int(r["voluntario"]) in train_vols else test_idx).append(i)
    return np.array(train_idx), np.array(test_idx)


def plot_confusion(cm, labels, out_png):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_title("View classification — Confusion matrix")
    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Retrain baseline with labeled views.csv")
    ap.add_argument("--root", required=True, help="Project root")
    args = ap.parse_args()
    ROOT = Path(args.root).resolve()
    META = ROOT / "metadata"
    META.mkdir(parents=True, exist_ok=True)

    # Load
    X, man, views = load_artifacts(ROOT)

    # Merge views onto manifest to bring pdf_base/vol/seq + preserve manifest row order
    man_small = man[["filepath", "pdf_base", "voluntario", "seq"]].copy()
    df = man_small.merge(views[["filepath", "view"]], on="filepath", how="left")

    # Keep only labeled rows
    df_labeled = df[~df["view"].isna()].copy().reset_index(drop=True)
    if df_labeled.empty:
        raise RuntimeError("No labeled samples found in metadata/views.csv.")

    # Select rows in X that correspond to these filepaths.
    # Since features.npy aligns with manifest rows, we create a boolean mask by filepath.
    idx_by_fp = {fp: i for i, fp in enumerate(man["filepath"].tolist())}
    idx_rows = []
    for fp in df_labeled["filepath"]:
        if fp not in idx_by_fp:
            raise KeyError(f"Filepath in views.csv not found in manifest.csv: {fp}")
        idx_rows.append(idx_by_fp[fp])
    idx_rows = np.array(idx_rows, dtype=int)

    Xsub = X[idx_rows]
    y = df_labeled["view"].values

    # Split by volunteer (per pdf)
    tr, te = split_by_volunteer(df_labeled)
    Xtr, Xte = Xsub[tr], Xsub[te]
    ytr, yte = y[tr], y[te]
    df_te = df_labeled.iloc[te].reset_index(drop=True)

    # Train baseline
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        PCA(n_components=128, random_state=42),
        LinearSVC(class_weight="balanced", C=1.0, random_state=42),
    )
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    # Report
    print("=== View Classification Report (test, volunteer-level split) ===")
    print(classification_report(yte, ypred, digits=3))

    # Confusion matrix
    labels = sorted(list(set(y)))
    cm = confusion_matrix(yte, ypred, labels=labels)
    out_png = META / "view_confusion_matrix.png"
    plot_confusion(cm, labels, out_png)
    print("Saved confusion matrix to", out_png)

    # Decision margins (use model decision_function on Xte transformed)
    Xte_tr = clf.named_steps["standardscaler"].transform(Xte)
    Xte_pca = clf.named_steps["pca"].transform(Xte_tr)
    dec = clf.named_steps["linearsvc"].decision_function(Xte_pca)

    margins = []
    if dec.ndim == 1:
        margins = np.abs(dec).astype(float)
    else:
        # multi-class: margin = s_true - max_other
        classes = clf.named_steps["linearsvc"].classes_.tolist()
        for i in range(dec.shape[0]):
            true = yte[i]
            tix = classes.index(true) if true in classes else None
            if tix is None:
                margins.append(0.0)
            else:
                s_true = dec[i, tix]
                s_other = np.max([dec[i, j] for j in range(dec.shape[1]) if j != tix])
                margins.append(float(max(s_true - s_other, 0.0)))
        margins = np.array(margins, dtype=float)

    # Save predictions CSV
    out_preds = META / "pred_view_results.csv"
    pd.DataFrame(
        {
            "filepath": df_te["filepath"],
            "y_true": yte,
            "y_pred": ypred,
            "margin": margins,
        }
    ).to_csv(out_preds, index=False)
    print("Saved predictions to", out_preds)


if __name__ == "__main__":
    main()
