#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt


def load_data(root: Path):
    X = np.load(root / "metadata/features.npy")
    man = pd.read_csv(root / "metadata/manifest.csv")
    views = pd.read_csv(root / "metadata/views.csv")  # needs columns: filepath, view
    df = man.merge(views[["filepath", "view"]], on="filepath", how="inner")
    return X, df


def split_by_volunteer(df):
    # simple per-pdf split: 70/30 by volunteer (stratified-ish)
    train_idx, test_idx = [], []
    for pdf, g in df.groupby("pdf_base"):
        vols = sorted(g["voluntario"].astype(int).unique())
        n = len(vols)
        n_train = max(1, int(0.7 * n))
        train_vols = set(vols[:n_train])
        test_vols = set(vols[n_train:])
        for i, r in g.iterrows():
            (train_idx if int(r["voluntario"]) in train_vols else test_idx).append(i)
    return np.array(train_idx), np.array(test_idx)


def plot_confusion(cm, labels, out_png):
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Baseline view classifier on handcrafted features."
    )
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    ROOT = Path(args.root)

    X, df = load_data(ROOT)
    y = df["view"].values
    files = df["filepath"].values

    # subset indices aligning with X rows
    idx = df.index.values
    Xsub = X[idx]

    # split by volunteer (to avoid leakage)
    tr, te = split_by_volunteer(df)
    Xtr, Xte = Xsub[tr], Xsub[te]
    ytr, yte = y[tr], y[te]

    # train simple logistic regression
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        PCA(n_components=128, random_state=42),
        LinearSVC(class_weight="balanced", C=1.0, random_state=42),
    )
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    # report
    print("=== View Classification Report (test, volunteer-level split) ===")
    print(classification_report(yte, ypred, digits=3))

    # confusion matrix
    labels = sorted(list(set(y)))
    lab_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(yte, ypred):
        cm[lab_to_idx[a], lab_to_idx[b]] += 1
    out_png = ROOT / "metadata/view_confusion_matrix.png"
    plot_confusion(cm, labels, out_png)
    print("Saved confusion matrix to", out_png)


if __name__ == "__main__":
    main()
