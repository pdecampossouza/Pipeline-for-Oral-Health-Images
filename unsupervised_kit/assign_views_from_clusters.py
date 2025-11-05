#!/usr/bin/env python3
import argparse, csv, os, shutil
from pathlib import Path


def parse_mapping(arglist):
    """
    Parse mapping like:
      --map "Pranchetas fotografias - dentição permanente - DAI treinamento:0=frontal,1=occlusal"
      --map "Pranchetas fotografias - dentição permanente - CPOD treinamento:0=frontal,1=occlusal"
    Returns dict: {pdf_base: {0:"frontal", 1:"occlusal"}}
    """
    mapping = {}
    for item in arglist or []:
        if ":" not in item:
            continue
        pdf, right = item.split(":", 1)
        pdf = pdf.strip()
        d = {}
        for pair in right.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                d[int(k.strip())] = v.strip()
        mapping[pdf] = d
    return mapping


def main():
    ap = argparse.ArgumentParser(
        description="Assign human-readable view names from KMeans clusters."
    )
    ap.add_argument(
        "--root", required=True, help="Dataset root (has metadata/ and images/)."
    )
    ap.add_argument(
        "--default_names",
        default="viewA,viewB",
        help="Fallback names for clusters 0,1 if no per-dataset mapping is provided.",
    )
    ap.add_argument(
        "--map",
        action="append",
        help="Per-dataset mapping: '<pdf_base>:0=frontal,1=occlusal' (repeatable).",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Also copy images into images_by_view/<pdf>/<view>/",
    )
    args = ap.parse_args()

    ROOT = Path(args.root)
    man_path = ROOT / "metadata/manifest.csv"
    clu_path = ROOT / "metadata/clusters.csv"
    out_csv = ROOT / "metadata/views.csv"

    # load data
    rows = list(csv.DictReader(open(man_path, newline="", encoding="utf-8")))
    clusters = {
        r["filepath"]: r["cluster"]
        for r in csv.DictReader(open(clu_path, newline="", encoding="utf-8"))
    }

    # mapping
    per_dataset = parse_mapping(args.map)
    default_names = [x.strip() for x in args.default_names.split(",")]
    if len(default_names) < 2:
        default_names = (default_names + ["viewB"])[:2]

    out = []
    for r in rows:
        pdf = r["pdf_base"]
        c = int(clusters.get(r["filepath"], -1))
        name_map = per_dataset.get(pdf, {})
        if c in name_map:
            view = name_map[c]
        else:
            # fallback: deterministic by cluster id
            view = default_names[c] if 0 <= c < len(default_names) else f"cluster_{c}"
        out.append(
            {
                "filepath": r["filepath"],
                "pdf_base": pdf,
                "volunteer": r["voluntario"],
                "seq": r["seq"],
                "cluster": c,
                "view": view,
            }
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["filepath", "pdf_base", "volunteer", "seq", "cluster", "view"],
        )
        w.writeheader()
        w.writerows(out)

    print("Saved views CSV to", out_csv)

    if args.copy:
        dest_root = ROOT / "images_by_view"
        if dest_root.exists():
            shutil.rmtree(dest_root)
        for r in out:
            src = ROOT / r["filepath"]
            d = dest_root / r["pdf_base"] / r["view"]
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, d / Path(r["filepath"]).name)
        print("Copied images to", dest_root)


if __name__ == "__main__":
    main()
