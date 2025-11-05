#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
import shutil, os

def load_manifest(root: Path):
    import csv
    rows = []
    with open(root/'metadata/manifest.csv', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Diretório raiz do dataset')
    ap.add_argument('--k', type=int, default=3, help='Número de clusters por pdf_base (ex.: vistas)')
    ap.add_argument('--copy', action='store_true', help='Copiar imagens para pastas clusters/<pdf_base>/cluster_i')
    args = ap.parse_args()
    ROOT = Path(args.root)
    feats = np.load(ROOT/'metadata/features.npy')
    rows = load_manifest(ROOT)

    # Agrupar por pdf_base
    by_pdf = {}
    for i, row in enumerate(rows):
        by_pdf.setdefault(row['pdf_base'], []).append(i)

    out_csv = ROOT/'metadata/clusters.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        import csv
        w = csv.writer(f)
        w.writerow(['filepath','pdf_base','voluntario','seq','cluster'])
        for pdf, idxs in by_pdf.items():
            X = feats[idxs]
            k = min(args.k, max(1, len(idxs)))
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
            labels = km.labels_.tolist()
            for j, i_global in enumerate(idxs):
                row = rows[i_global]
                row_cluster = labels[j]
                w.writerow([row['filepath'], row['pdf_base'], row['voluntario'], row['seq'], row_cluster])

    print('Clusters salvos em', out_csv)

    if args.copy:
        # copiar para ROOT/clusters/<pdf_base>/cluster_X/arquivo.jpg
        dest_root = ROOT/'clusters'
        if dest_root.exists():
            shutil.rmtree(dest_root)
        dest_root.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out_csv, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                src = ROOT/row['filepath']
                d = dest_root/row['pdf_base']/f"cluster_{row['cluster']}"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, d/src.name)
        print('Cópias por cluster em', dest_root)

if __name__ == '__main__':
    main()
