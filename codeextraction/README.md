# CodeExtraction — SBBrasil TrainSheets
Utilities to extract images from PDF modules, assign volunteer IDs, rename/copy into a clean structure,
and generate a manifest CSV compatible with the rest of the pipeline.

## Files
- `extract_and_catalog.py` — one-stop script: extracts images, infers volunteer markers (`Voluntário N`) from PDF text,
  renames and copies images to `images/<pdf_base>/Voluntario<id>/...`, and writes `metadata/manifest.csv`.
- `verify_images.py` — quick sanity checks (size thresholds, logo-like tiny images, duplicates by hash).
- `requirements.txt` — minimal dependencies.

## Quick start
```bash
# Example (PowerShell / CMD)
python codeextraction\extract_and_catalog.py --root "." --pdf_dir "pdfs" --min_kb 30

# Options:
#  --pdf_dir         Folder containing the PDF modules (default: ./pdfs)
#  --out_images      Output base folder for images (default: ./images)
#  --meta_dir        Output metadata folder (default: ./metadata)
#  --min_kb          Discard images below this size in kilobytes (default: 30)
#  --render_if_missing  Render full page (300 DPI) when a page has no embedded rasters (default: off)
#  --assign_window   Pages to look around for 'Voluntário <N>' markers (default: +3, -1)
#  --pdf_glob        Pattern for PDFs (default: *.pdf)
```

This script is conservative and transparent:
- Every saved image is recorded in the manifest with: pdf_base, voluntario, seq, page, width, height, bytes, source.
- Logos/footers are typically filtered by `--min_kb 30` and `min(width,height) >= 200` heuristics.
- Volunteer assignment uses text markers in nearby pages; when none is found, images are assigned to the **current** volunteer bucket (carried over) and flagged `vol_inferred=1`.
