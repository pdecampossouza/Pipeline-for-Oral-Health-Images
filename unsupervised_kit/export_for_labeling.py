# -*- coding: utf-8 -*-
"""
Export a review set for interactive labeling.
Creates metadata/review_candidates.csv with:
[filepath, pdf_base, voluntario, seq, y_pred, margin, y_true]

Usage:
  python unsupervised_kit/export_for_labeling.py --root "." --only_uncertain 200 --copy
"""

import argparse
from pathlib import Path
import shutil
import pandas as pd
import numpy as np


TARGET_PDFS_DEFAULT = [
    # leave empty to include ALL; or put exact names if you want to restrict
    # "Pranchetas fotografias - dentição permanente - DAI treinamento",
    # "Pranchetas fotografias - dentição permanente - CPOD treinamento",
]


def normalize_paths(df: pd.DataFrame, col="filepath") -> pd.DataFrame:
    df[col] = df[col].astype(str).str.replace("\\\\", "\\").str.replace("/", "\\")
    return df


def main():
    ap = argparse.ArgumentParser(description="Prepare candidates for manual labeling.")
    ap.add_argument(
        "--root", required=True, help="Project root (has metadata/ and images)"
    )
    ap.add_argument(
        "--only_uncertain",
        type=int,
        default=0,
        help="If >0, keep only N most-uncertain (lowest margin) samples.",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Also copy candidate images under review/{pred}/ for quick browsing.",
    )
    ap.add_argument(
        "--pdf_filter",
        nargs="*",
        default=TARGET_PDFS_DEFAULT,
        help="Optional list of pdf_base names to include. If empty, include all.",
    )
    args = ap.parse_args()

    ROOT = Path(args.root).resolve()
    META = ROOT / "metadata"
    META.mkdir(parents=True, exist_ok=True)

    # Load manifest (source of truth for pdf_base/vol/seq)
    man_path = META / "manifest.csv"
    if not man_path.exists():
        raise FileNotFoundError(f"Missing {man_path}")
    man = pd.read_csv(man_path)
    required_mcols = {"filepath", "pdf_base", "voluntario", "seq"}
    missing_m = required_mcols - set(man.columns)
    if missing_m:
        raise ValueError(f"manifest.csv missing columns: {missing_m}")
    man = normalize_paths(man, "filepath")
    man_small = man[["filepath", "pdf_base", "voluntario", "seq"]].copy()

    # Try predictions first
    preds_path = META / "pred_view_results.csv"
    if preds_path.exists():
        dfp = pd.read_csv(preds_path)
        dfp = normalize_paths(dfp, "filepath")
        # ensure columns
        if "y_pred" not in dfp.columns:
            raise ValueError("pred_view_results.csv must contain column 'y_pred'")
        if "margin" not in dfp.columns:
            dfp["margin"] = np.nan
        if "y_true" not in dfp.columns:
            dfp["y_true"] = ""
        base = dfp[["filepath", "y_pred", "margin", "y_true"]].copy()
        source = "pred_view_results.csv"
    else:
        # fall back to views.csv (no predictions)
        views_path = META / "views.csv"
        if not views_path.exists():
            raise FileNotFoundError(
                "Neither metadata/pred_view_results.csv nor metadata/views.csv found. "
                "Run the baseline trainer first to create pred_view_results.csv."
            )
        v = pd.read_csv(views_path)
        v = normalize_paths(v, "filepath")
        base = v.rename(columns={"view": "y_true"})[["filepath", "y_true"]].copy()
        base["y_pred"] = ""
        base["margin"] = np.nan
        source = "views.csv"

    # Merge to inject pdf_base / voluntario / seq
    df = base.merge(man_small, on="filepath", how="left")
    # Keep only existing files
    df["exists"] = df["filepath"].apply(lambda p: (ROOT / p).exists())
    df = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)

    # Optional: filter by selected pdf_base names
    if args.pdf_filter:
        df = df[df["pdf_base"].isin(args.pdf_filter)].reset_index(drop=True)

    # De-duplicate by filepath
    df = df.drop_duplicates(subset=["filepath"]).reset_index(drop=True)

    # Keep only N most uncertain if requested
    if args.only_uncertain and "margin" in df.columns:
        # lower margin = more uncertain; NaN goes to the end
        df = df.sort_values(["margin"], ascending=True, na_position="last")
        df = df.head(int(args.only_uncertain)).reset_index(drop=True)

    out_csv = META / "review_candidates.csv"
    df[
        ["filepath", "pdf_base", "voluntario", "seq", "y_pred", "margin", "y_true"]
    ].to_csv(out_csv, index=False, quoting=1)
    print(f"Saved review candidates: {out_csv}  | source={source}  | rows={len(df)}")

    if args.copy:
        dst_root = ROOT / "review"
        if dst_root.exists():
            shutil.rmtree(dst_root)
        dst_root.mkdir(parents=True, exist_ok=True)
        for _, r in df.iterrows():
            pred = str(r.get("y_pred") or "unlabeled")
            sub = dst_root / pred
            sub.mkdir(parents=True, exist_ok=True)
            src = ROOT / r["filepath"]
            # create a readable name
            name = f"{Path(r['pdf_base']).name if isinstance(r['pdf_base'], str) else 'pdf'}__vol{r.get('voluntario','')}__seq{r.get('seq','')}__{Path(r['filepath']).name}"
            dst = sub / name
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass
        print(f"Copied {len(df)} images under {dst_root}")


if __name__ == "__main__":
    main()
