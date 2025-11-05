#!/usr/bin/env python3
import argparse, csv
from pathlib import Path

HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Volunteer Gallery</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{ --bg:#0b1320; --card:#111a2e; --ink:#e9eef9; --muted:#a9b7d0; --accent:#6aa9ff; }
*{box-sizing:border-box}
body{margin:24px; font-family:Inter,system-ui,Segoe UI,Roboto,Arial,sans-serif; background:var(--bg); color:var(--ink)}
h1{font-size:28px; margin:0 0 8px}
h2{margin:32px 0 12px; font-size:22px; color:var(--accent)}
.section{margin-bottom:28px; border-top:1px solid #1f2942; padding-top:18px}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px}
.card{background:var(--card); border:1px solid #1f2942; border-radius:14px; padding:12px; box-shadow:0 6px 20px rgba(0,0,0,0.35)}
.card h3{margin:0 0 8px; font-size:16px; color:var(--ink)}
.thumb{display:block; border-radius:10px; overflow:hidden; border:1px solid #2a3557}
.thumb img{width:100%; height:180px; object-fit:contain; background:#0a0f1e}
.list{margin-top:8px; font-size:12px; color:var(--muted)}
.list a{color:var(--muted); text-decoration:none}
.list a:hover{color:var(--accent)}
.header{display:flex; align-items:center; gap:10px; justify-content:space-between}
.badge{font-size:12px; background:#0e1b34; padding:2px 8px; border-radius:999px; border:1px solid #1f2942; color:#b7c6e6}
</style>
</head>
<body>
<div class="header">
  <h1>Volunteer Gallery</h1>
  <span class="badge">generated from metadata/manifest.csv</span>
</div>
<p style="color:#a9b7d0;margin-bottom:20px">
Click any thumbnail to open the full-size image. Each section below corresponds to a dataset (PDF_BASE).
</p>
"""
HTML_TAIL = """</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", required=True, help="Dataset root (has images/ and metadata/)."
    )
    args = ap.parse_args()
    ROOT = Path(args.root)
    man = ROOT / "metadata/manifest.csv"

    rows = []
    with open(man, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)

    from collections import defaultdict

    per_pdf_vol = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_pdf_vol[r["pdf_base"]][int(r["voluntario"])].append(r["filepath"])

    out = ROOT / "metadata/gallery.html"
    parts = [HTML_HEAD]
    for pdf_base in sorted(per_pdf_vol.keys(), key=str.lower):
        parts.append(f'<div class="section"><h2>{pdf_base}</h2><div class="grid">')
        for vol in sorted(per_pdf_vol[pdf_base].keys()):
            files = sorted(per_pdf_vol[pdf_base][vol])
            parts.append('<div class="card">')
            parts.append(f"<h3>Volunteer {vol} · {len(files)} photos</h3>")
            rel0 = ("../" + files[0]).replace("\\", "/")
            parts.append(
                f'<a class="thumb" href="{rel0}" target="_blank"><img src="{rel0}" alt="Vol {vol}"></a>'
            )
            parts.append('<div class="list">')
            for rel in files:
                relw = ("../" + rel).replace("\\", "/")
                name = Path(rel).name
                parts.append(f'• <a href="{relw}" target="_blank">{name}</a><br>')
            parts.append("</div></div>")
        parts.append("</div></div>")
    parts.append(HTML_TAIL)
    out.write_text("\n".join(parts), encoding="utf-8")
    print("Gallery saved to", out)


if __name__ == "__main__":
    main()
