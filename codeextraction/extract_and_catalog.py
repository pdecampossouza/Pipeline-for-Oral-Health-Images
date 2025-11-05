#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract images from PDF modules, infer volunteer IDs from text markers, and build a manifest.
Outputs a clean folder structure:
  images/<pdf_base>/Voluntario<id>/<pdfslug>_Voluntario<id>_<seq>.jpg
and a CSV:
  metadata/manifest.csv (filepath, pdf_base, voluntario, seq, page, idx, width, height, bytes, src, md5)

Usage (Windows PowerShell / CMD):
  python codeextraction\extract_and_catalog.py --root "." --pdf_dir "pdfs" --min_kb 30
"""

import argparse, io, os, re, hashlib
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import pandas as pd
import numpy as np

VOL_RGX = re.compile(r"Volunt[áa]rio\s*([0-9]+)", re.IGNORECASE)

def md5_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.md5(b).hexdigest()

def slugify(s: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = s.strip().replace("  ", " ")
    return s

def page_text_markers(doc: fitz.Document):
    """Return a dict page->list of volunteer ids found on that page."""
    markers = {}
    for pno in range(doc.page_count):
        try:
            txt = doc.load_page(pno).get_text("text") or ""
        except Exception:
            txt = ""
        hits = []
        for m in VOL_RGX.finditer(txt):
            try:
                hits.append(int(m.group(1)))
            except Exception:
                pass
        if hits:
            markers[pno] = hits
    return markers

def nearest_volunteer_for_page(pno: int, markers: dict, back: int = 1, fwd: int = 3):
    """Search around pno for 'Voluntário N' markers. Prefer forward pages (common pattern)."""
    # same page
    if pno in markers and markers[pno]:
        return markers[pno][0]
    # forward first
    for k in range(1, fwd+1):
        q = pno + k
        if q in markers and markers[q]:
            return markers[q][0]
    # then backward
    for k in range(1, back+1):
        q = pno - k
        if q in markers and markers[q]:
            return markers[q][0]
    return None

def save_jpeg(img_bytes: bytes, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(img_bytes)) as im:
        rgb = im.convert("RGB")
        rgb.save(out_path, format="JPEG", quality=95, optimize=True, progressive=True)

def render_page(doc: fitz.Document, pno: int, zoom: float = 300/72):
    page = doc.load_page(pno)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def main():
    ap = argparse.ArgumentParser(description="Extract and catalog images from PDF modules.")
    ap.add_argument("--root", default=".", help="Project root (outputs under this folder)")
    ap.add_argument("--pdf_dir", default="pdfs", help="Folder with PDF files")
    ap.add_argument("--out_images", default="images", help="Images output root")
    ap.add_argument("--meta_dir", default="metadata", help="Metadata output folder")
    ap.add_argument("--pdf_glob", default="*.pdf", help="PDF filename pattern")
    ap.add_argument("--min_kb", type=int, default=30, help="Skip images smaller than this (KB)")
    ap.add_argument("--min_side", type=int, default=200, help="Skip if min(width,height) < min_side")
    ap.add_argument("--assign_window", type=str, default="+3,-1", help="Pages lookahead,lookbehind for volunteer markers")
    ap.add_argument("--render_if_missing", action="store_true", help="Render full page (300DPI) when no rasters are embedded")
    args = ap.parse_args()

    ROOT = Path(args.root).resolve()
    PDF_DIR = (ROOT / args.pdf_dir).resolve()
    OUT_IMAGES = (ROOT / args.out_images).resolve()
    META = (ROOT / args.meta_dir).resolve()
    META.mkdir(parents=True, exist_ok=True)

    fwd, back = 3, 1
    try:
        fwd, back = [int(x) for x in args.assign_window.split(",")]
    except Exception:
        pass

    rows = []
    for pdf_path in sorted(PDF_DIR.glob(args.pdf_glob)):
        pdf_base = pdf_path.stem  # include full name; keep accents
        pdf_slug = slugify(pdf_base)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"[WARN] Could not open {pdf_path}: {e}")
            continue

        markers = page_text_markers(doc)
        current_vol = 1
        seq_per_vol = {}

        # Extract embedded rasters, preserving page order
        any_saved = False
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            imgs = page.get_images(full=True)  # list of tuples (xref, ...)
            if not imgs and args.render_if_missing:
                # render fallback
                im = render_page(doc, pno)
                w, h = im.size
                if min(w, h) >= args.min_side:
                    with io.BytesIO() as buf:
                        im.save(buf, format="JPEG", quality=90)
                        b = buf.getvalue()
                    if len(b) >= args.min_kb * 1024:
                        vol = nearest_volunteer_for_page(pno, markers, back=back, fwd=fwd)
                        vol_inferred = 0
                        if vol is None:
                            vol = current_vol
                            vol_inferred = 1
                        current_vol = vol
                        seq_per_vol[vol] = seq_per_vol.get(vol, 0) + 1
                        seq = seq_per_vol[vol]

                        fname = f"{pdf_slug}_Voluntario{vol}_{seq:02d}.jpg"
                        rel = Path(args.out_images) / pdf_base / f"Voluntario{vol}" / fname
                        save_jpeg(b, ROOT / rel)
                        rows.append({
                            "filepath": str(rel).replace("\\","/"),
                            "pdf_base": pdf_base,
                            "voluntario": vol,
                            "seq": seq,
                            "page": pno+1,
                            "idx": -1,
                            "width": w,
                            "height": h,
                            "bytes": len(b),
                            "src": "rendered",
                            "md5": md5_bytes(b),
                            "vol_inferred": vol_inferred,
                        })
                        any_saved = True
                continue

            for idx, info in enumerate(imgs):
                xref = info[0]
                try:
                    d = doc.extract_image(xref)
                except Exception:
                    continue
                b = d.get("image", b"")
                w, h = d.get("width", 0), d.get("height", 0)
                if not b or len(b) < args.min_kb * 1024 or min(w, h) < args.min_side:
                    continue

                vol = nearest_volunteer_for_page(pno, markers, back=back, fwd=fwd)
                vol_inferred = 0
                if vol is None:
                    vol = current_vol
                    vol_inferred = 1
                current_vol = vol

                seq_per_vol[vol] = seq_per_vol.get(vol, 0) + 1
                seq = seq_per_vol[vol]

                fname = f"{pdf_slug}_Voluntario{vol}_{seq:02d}.jpg"
                rel = Path(args.out_images) / pdf_base / f"Voluntario{vol}" / fname

                # Convert to JPEG bytes if needed
                try:
                    img = Image.open(io.BytesIO(b)).convert("RGB")
                    out = io.BytesIO()
                    img.save(out, format="JPEG", quality=95, optimize=True, progressive=True)
                    jb = out.getvalue()
                except Exception:
                    jb = b  # if already JPEG

                save_jpeg(jb, ROOT / rel)
                rows.append({
                    "filepath": str(rel).replace("\\","/"),
                    "pdf_base": pdf_base,
                    "voluntario": vol,
                    "seq": seq,
                    "page": pno+1,
                    "idx": idx,
                    "width": w,
                    "height": h,
                    "bytes": len(jb),
                    "src": "embedded",
                    "md5": md5_bytes(jb),
                    "vol_inferred": vol_inferred,
                })
                any_saved = True

        doc.close()
        if not any_saved:
            print(f"[INFO] No images extracted from {pdf_path} (consider --render_if_missing)")

    # Build manifest (stable order by pdf_base, voluntario, seq)
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["pdf_base","voluntario","seq","page","idx"]).reset_index(drop=True)
        META.mkdir(parents=True, exist_ok=True)
        out_csv = META / "manifest.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved manifest: {out_csv} | rows={len(df)}")
    else:
        print("No images collected. Check thresholds (min_kb/min_side) or try --render_if_missing.")

if __name__ == "__main__":
    main()
