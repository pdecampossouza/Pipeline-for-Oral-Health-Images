# -*- coding: utf-8 -*-
# Visualize 2D PCA of features colored by cluster assignments.
# Colors are fixed: cluster 0 = blue, cluster 1 = red, -1 (unassigned) = gray.
#
# Usage:
#   python unsupervised_kit/visualize_clusters.py --root "." [--per_pdf]
#
# Outputs:
#   metadata/clusters_pca.png
#   metadata/clusters_pca__<pdf_base>.png  (when --per_pdf is on)

from pathlib import Path
import argparse
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches


# ---------- Fixed, high-contrast palette ----------
CLUSTER_COLOR_MAP: Dict[int, str] = {
    -1: "#9ca3af",  # gray (unassigned / missing)
    0: "#1f77b4",  # blue
    1: "#d62728",  # red
    2: "#2ca02c",  # green (future-proof if k>2)
    3: "#9467bd",  # purple
}


def colors_for_labels(labels: np.ndarray) -> List[str]:
    """Map each label to a hex color."""
    labels = np.asarray(labels).astype(int)
    return [CLUSTER_COLOR_MAP.get(int(l), "#9ca3af") for l in labels]


def sanitize_filename(s: str) -> str:
    """Make a filesystem-friendly filename for per-dataset figures."""
    s = s.strip()
    # keep ASCII letters, numbers, spaces, dashes and underscores; replace the rest with '_'
    s = re.sub(r"[^a-zA-Z0-9\-\_\ \.]", "_", s)
    s = s.replace("/", "_").replace("\\", "_")
    return s


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
        help="Project root (contains metadata/ and image folders)",
    )
    ap.add_argument(
        "--per_pdf",
        action="store_true",
        help="Also save one PCA figure per PDF module (pdf_base)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    ROOT = Path(args.root).resolve()
    META = ROOT / "metadata"
    META.mkdir(exist_ok=True, parents=True)

    # ---- Load artifacts
    X = np.load(META / "features.npy")  # (N, D)
    man = pd.read_csv(
        META / "manifest.csv"
    )  # must have 'filepath','pdf_base', 'voluntario','seq' (seq optional)
    if not (META / "clusters.csv").exists():
        raise FileNotFoundError(
            "metadata/clusters.csv not found. Run cluster assignment first."
        )

    cls = pd.read_csv(
        META / "clusters.csv"
    )  # must have 'filepath' and 'cluster' (or similar)
    # robust detection of the cluster column
    cluster_col = None
    for cand in ["cluster", "clusters", "label", "kmeans_label"]:
        if cand in cls.columns:
            cluster_col = cand
            break
    if cluster_col is None:
        raise ValueError(
            "clusters.csv must contain a 'cluster' (or equivalent) column."
        )

    # Align to manifest order (features.npy is aligned to manifest rows)
    df = man.merge(cls[["filepath", cluster_col]], on="filepath", how="left")
    if len(df) != len(X):
        raise ValueError(
            f"Row mismatch: features ({len(X)}) vs manifest+clusters ({len(df)})."
        )

    labels = df[cluster_col].fillna(-1).astype(int).to_numpy()
    pdf_base = df["pdf_base"].astype(str).to_numpy()

    # ---- PCA on full set (2D for plotting)
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)

    # ---- Global figure
    plt.figure(figsize=(9, 6))
    plt.scatter(X2[:, 0], X2[:, 1], s=18, c=colors_for_labels(labels))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of features — colored by cluster")

    # legend from unique labels
    uniq = sorted(set(labels.tolist()))
    handles = []
    for lab in uniq:
        name = "unassigned" if lab == -1 else f"cluster {lab}"
        handles.append(
            mpatches.Patch(color=CLUSTER_COLOR_MAP.get(int(lab), "#9ca3af"), label=name)
        )
    if handles:
        plt.legend(handles=handles, loc="upper right", frameon=False)

    out_global = META / "clusters_pca.png"
    plt.tight_layout()
    plt.savefig(out_global, dpi=150)
    plt.close()
    print(f"Saved global PCA figure to {out_global}")

    # ---- Per-dataset figures (optional)
    if args.per_pdf:
        unique_pdfs = df["pdf_base"].unique().tolist()
        for pb in unique_pdfs:
            mask = (df["pdf_base"] == pb).to_numpy()
            if not mask.any():
                continue
            X2_sub = X2[mask]
            labels_sub = labels[mask]

            plt.figure(figsize=(9, 6))
            plt.scatter(
                X2_sub[:, 0], X2_sub[:, 1], s=20, c=colors_for_labels(labels_sub)
            )
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(pb)

            uniq_sub = sorted(set(labels_sub.tolist()))
            handles = []
            for lab in uniq_sub:
                name = "unassigned" if lab == -1 else f"cluster {lab}"
                handles.append(
                    mpatches.Patch(
                        color=CLUSTER_COLOR_MAP.get(int(lab), "#9ca3af"), label=name
                    )
                )
            if handles:
                plt.legend(handles=handles, loc="upper right", frameon=False)

            fname = f"clusters_pca__{sanitize_filename(pb)}.png"
            out_path = META / fname
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved per-dataset PCA: {out_path}")


if __name__ == "__main__":
    main()
