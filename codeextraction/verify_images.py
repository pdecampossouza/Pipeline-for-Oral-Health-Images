#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity checks for the extracted dataset.
- Reports tiny images (logo/footers), duplicates by MD5, and missing files from manifest.
- Optional cleanup suggestions.

Usage:
  python codeextraction\verify_images.py --root "."
"""
from pathlib import Path
import argparse, hashlib
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Verify extracted images and manifest.")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--min_kb", type=int, default=30)
    ap.add_argument("--min_side", type=int, default=200)
    args = ap.parse_args()

    ROOT = Path(args.root).resolve()
    man = pd.read_csv(ROOT / "metadata/manifest.csv")
    print(f"Manifest rows: {len(man)}")

    # existence check
    man["exists"] = man["filepath"].apply(lambda p: (ROOT / p).exists())
    missing = man[~man["exists"]]
    if not missing.empty:
        print(f"[WARN] Missing files: {len(missing)}")
        print(missing[["filepath","pdf_base","voluntario","seq"]].head(10).to_string(index=False))

    # small images
    small = man[(man["bytes"] < args.min_kb*1024) | (man[["width","height"]].min(axis=1) < args.min_side)]
    if not small.empty:
        print(f"[INFO] Small images (likely logos/footers): {len(small)}")
        print(small[["filepath","bytes","width","height"]].head(10).to_string(index=False))

    # duplicates by md5
    dups = man.groupby("md5").size().reset_index(name="count")
    dups = dups[dups["count"] > 1]
    if not dups.empty:
        print(f"[INFO] Duplicate image hashes: {len(dups)} groups")
        # show sample group
        sample = man.merge(dups, on="md5")
        print(sample.sort_values("md5").head(12)[["md5","filepath"]].to_string(index=False))

    print("Verification done.")

if __name__ == "__main__":
    main()
