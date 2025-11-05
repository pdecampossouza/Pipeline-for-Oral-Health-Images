#!/usr/bin/env python3
import argparse, os, csv
from pathlib import Path
import numpy as np
from PIL import Image

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def build_index(root: Path):
    feats = np.load(root/'metadata/features.npy')
    rows = []
    with open(root/'metadata/manifest.csv', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return feats, rows

def compute_feature(fp: Path, size=224):
    from PIL import Image
    import numpy as np
    with Image.open(fp) as im:
        im = im.convert('RGB').resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    gray = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2])
    gx = np.gradient(gray, axis=1); gy = np.gradient(gray, axis=0)
    edge = np.hypot(gx, gy)
    def down(a, f=4):
        h, w = a.shape; hh, ww = h//f, w//f
        return a[:hh*f, :ww*f].reshape(hh, f, ww, f).mean(axis=(1,3))
    g56 = down(gray, 4).astype(np.float32)
    e56 = down(edge, 4).astype(np.float32)
    return np.concatenate([g56.flatten(), e56.flatten()])

def make_html(out_html: Path, query_rel: str, results):
    rows = []
    rows.append('<html><head><meta charset="utf-8"><style>img{max-width:240px} .row{display:flex;gap:12px;margin:8px 0}</style></head><body>')
    rows.append(f'<h3>Consulta: {query_rel}</h3>')
    rows.append('<div class="row">')
    rows.append(f'<div><img src="{query_rel}"><div>consulta</div></div>')
    rows.append('</div><hr>')
    rows.append('<h3>Top vizinhos</h3>')
    rows.append('<div class="row">')
    for rel, score in results:
        rows.append(f'<div><img src="{rel}"><div>score={score:.3f}<br>{rel}</div></div>')
    rows.append('</div></body></html>')
    out_html.write_text("\n".join(rows), encoding='utf-8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Diretório raiz do dataset (onde foi rodado features.py)')
    ap.add_argument('--query', required=True, help='Caminho da imagem de consulta (absoluto ou relativo ao ROOT)')
    ap.add_argument('--topk', type=int, default=12)
    args = ap.parse_args()

    ROOT = Path(args.root)
    feats, rows = build_index(ROOT)
    # normaliza caminhos
    files = [str((ROOT/row['filepath']).resolve()) for row in rows]

    qpath = Path(args.query)
    if not qpath.is_file():
        # tentar relativo ao ROOT
        qpath = ROOT/args.query
    qfeat = compute_feature(qpath)

    # Similaridade
    sims = [cosine_sim(qfeat, feats[i]) for i in range(len(files))]
    top_idx = np.argsort(sims)[::-1][:args.topk]
    results = [(rows[i]['filepath'], sims[i]) for i in top_idx]

    # HTML de saída
    out_html = ROOT/'metadata'/'nn_search.html'
    # produzir caminhos relativos ao HTML
    rel_results = []
    for rel, score in results:
        rel_results.append((rel.replace('\\','/'), score))
    query_rel = os.path.relpath(str(qpath), str(ROOT)).replace('\\','/')
    make_html(out_html, query_rel, rel_results)
    print('Resultado salvo em', out_html)

if __name__ == '__main__':
    main()
