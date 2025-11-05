#!/usr/bin/env python3
import os, re, argparse, csv
from pathlib import Path
import numpy as np
from PIL import Image

name_rx = re.compile(r'^(?P<pdf_base>.+)_Voluntario(?P<vol>\d+)(?:_(?P<seq>\d+))?\.(?P<ext>jpg|jpeg|png)$', re.I)
IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}

def collect_images(images_dir: Path):
    items = []
    for p in images_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            m = name_rx.match(p.name)
            if not m: 
                continue
            items.append((p, m.group('pdf_base'), int(m.group('vol')), int(m.group('seq') or 1)))
    items.sort(key=lambda x: (x[1], x[2], x[3]))
    return items

def compute_feature(fp: Path, size=224):
    # Feature simples e robusta: grayscale reamostrado + mapa de bordas por gradiente
    with Image.open(fp) as im:
        im = im.convert('RGB').resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    gray = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2])
    # Sobel-like via gradiente numpy
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    edge = np.hypot(gx, gy)
    # downsample para 56x56 para compactar
    def down(a, f=4):
        h, w = a.shape
        hh, ww = h//f, w//f
        return a[:hh*f, :ww*f].reshape(hh, f, ww, f).mean(axis=(1,3))
    g56 = down(gray, 4).astype(np.float32)
    e56 = down(edge, 4).astype(np.float32)
    feat = np.concatenate([g56.flatten(), e56.flatten()])  # 2*56*56 = 6272 dims
    return feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Diretório raiz do dataset (contém pastas de imagens)')
    ap.add_argument('--images_subdir', default='images', help='Subpasta com as imagens (default: images)')
    args = ap.parse_args()
    ROOT = Path(args.root)
    IMAGES = ROOT/args.images_subdir
    OUT = ROOT/'metadata'
    OUT.mkdir(parents=True, exist_ok=True)

    items = collect_images(IMAGES)
    feats = []
    rows = []
    for p, pdf_base, vol, seq in items:
        feat = compute_feature(p)
        feats.append(feat)
        rows.append({
            'filepath': str(p.relative_to(ROOT)).replace('\\','/'),
            'pdf_base': pdf_base,
            'voluntario': vol,
            'seq': seq,
            'bytes': p.stat().st_size
        })
    feats = np.stack(feats, axis=0) if feats else np.zeros((0,6272), dtype=np.float32)
    np.save(OUT/'features.npy', feats)

    man_path = OUT/'manifest.csv'
    with open(man_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['filepath','pdf_base','voluntario','seq','bytes'])
        w.writeheader()
        w.writerows(rows)
    print('Salvo:', OUT/'features.npy', 'e', man_path, '| shape:', feats.shape)

if __name__ == '__main__':
    main()
