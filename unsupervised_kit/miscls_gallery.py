#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

HTML_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Misclassified Samples</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{ --bg:#0b1320; --card:#111a2e; --ink:#e9eef9; --muted:#a9b7d0; --accent:#6aa9ff; }
*{box-sizing:border-box}
body{margin:24px;font-family:Inter,system-ui,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--ink)}
h1{margin:0 0 8px;font-size:28px}
h2{margin:22px 0 8px;font-size:20px;color:var(--accent)}
.card{background:var(--card);border:1px solid #1f2942;border-radius:14px;padding:12px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.item{border:1px solid #243257;border-radius:10px;padding:8px}
.item img{width:100%;height:140px;object-fit:contain;background:#0a0f1e;border-radius:8px}
.meta{font-size:12px;color:var(--muted);margin-top:6px}
.badge{font-size:12px;background:#0e1b34;padding:2px 8px;border-radius:999px;border:1px solid #1f2942;color:#b7c6e6}
.kv{font-size:13px;color:#bcd}
hr{border:none;border-top:1px solid #1f2942;margin:18px 0}
</style></head><body>
"""

HTML_TAIL = "</body></html>"


def split_by_volunteer_stratified(df):
    """
    70/30 split by volunteer, roughly stratified by 'view' within each pdf_base.
    """
    import random

    random.seed(42)
    train_idx, test_idx = [], []
    for pdf, g in df.groupby("pdf_base"):
        by_view = {}
        for v, gg in g.groupby("view"):
            vols = sorted(gg["voluntario"].astype(int).unique())
            random.shuffle(vols)
            n = len(vols)
            n_train = max(1, int(0.7 * n))
            by_view[v] = {"train": set(vols[:n_train]), "test": set(vols[n_train:])}
        for i, r in g.iterrows():
            v = r["view"]
            vol = int(r["voluntario"])
            (train_idx if vol in by_view[v]["train"] else test_idx).append(i)
    return np.array(train_idx), np.array(test_idx)


def main():
    ap = argparse.ArgumentParser(
        description="Create an HTML gallery of misclassified test samples."
    )
    ap.add_argument(
        "--root", required=True, help="Dataset root (has metadata/ and images)."
    )
    ap.add_argument(
        "--pca_dims", type=int, default=128, help="PCA components (default: 128)"
    )
    ap.add_argument("--C", type=float, default=1.0, help="LinearSVC C (default: 1.0)")
    ap.add_argument(
        "--pdf_regex",
        type=str,
        default="",
        help="Filter rows by regex on pdf_base (optional)",
    )
    ap.add_argument(
        "--top_per_pair",
        type=int,
        default=40,
        help="Max samples per true→pred pair in HTML",
    )
    args = ap.parse_args()

    ROOT = Path(args.root)
    X = np.load(ROOT / "metadata/features.npy")
    man = pd.read_csv(ROOT / "metadata/manifest.csv")
    views = pd.read_csv(ROOT / "metadata/views.csv")
    df = man.merge(views[["filepath", "view"]], on="filepath", how="inner").copy()

    # optional filter by dataset
    if args.pdf_regex:
        rgx = re.compile(args.pdf_regex)
        df = df[df["pdf_base"].apply(lambda s: bool(rgx.search(str(s))))].copy()

    # Align X rows
    idx = df.index.values
    Xsub = X[idx]
    y = df["view"].values
    files = df["filepath"].values

    # split (volunteer-level, stratified)
    tr, te = split_by_volunteer_stratified(df)
    Xtr, Xte = Xsub[tr], Xsub[te]
    ytr, yte = y[tr], y[te]
    df_te = df.iloc[te].reset_index(drop=True)

    # model
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        PCA(n_components=args.pca_dims, random_state=42),
        LinearSVC(class_weight="balanced", C=args.C, random_state=42),
    )
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    # decision function for margins
    # (LinearSVC has decision_function; for binary returns margin; for multi returns per-class scores)
    decision_scores = clf.named_steps["linearsvc"].decision_function(
        clf.named_steps["pca"].transform(
            clf.named_steps["standardscaler"].transform(Xte)
        )
    )

    def margin_fn(i):
        s = decision_scores[i]
        try:
            # binary: float
            return float(abs(s))
        except Exception:
            # multi: max(score_true - max_other, 0)
            lablist = sorted(list(set(ytr)))
            true = yte[i]
            if true in lablist:
                tix = lablist.index(true)
                s_true = s[tix]
                s_other = np.max([s[j] for j in range(len(lablist)) if j != tix])
                return float(max(s_true - s_other, 0.0))
            return 0.0

    # report
    print("=== Misclassification Audit (test set) ===")
    print(classification_report(yte, ypred, digits=3))
    labels = sorted(list(set(y)))
    cm = confusion_matrix(yte, ypred, labels=labels)
    print("Confusion matrix (labels={}):\n{}".format(labels, cm))

    # collect misclassified
    mis = []
    for i in range(len(yte)):
        if yte[i] != ypred[i]:
            mis.append(
                {
                    "filepath": df_te.loc[i, "filepath"],
                    "pdf_base": df_te.loc[i, "pdf_base"],
                    "volunteer": df_te.loc[i, "voluntario"],
                    "seq": df_te.loc[i, "seq"],
                    "true": yte[i],
                    "pred": ypred[i],
                    "margin": margin_fn(i),
                }
            )
    mis_df = pd.DataFrame(mis).sort_values(
        ["true", "pred", "margin"], ascending=[True, True, True]
    )

    # save CSV
    out_csv = ROOT / "metadata/misclassified.csv"
    mis_df.to_csv(out_csv, index=False)
    print("Saved CSV:", out_csv)

    # HTML
    out_html = ROOT / "metadata/misclassified.html"
    parts = [
        HTML_HEAD,
        f"<div class='kv'><span class='badge'>pca_dims={args.pca_dims} · C={args.C} · pdf_regex='{args.pdf_regex or 'ALL'}'</span></div><hr>",
    ]
    # summary
    parts.append("<div class='card'><h2>Summary</h2>")
    parts.append("<pre>")
    parts.append(classification_report(yte, ypred, digits=3))
    parts.append("</pre>")
    parts.append("</div>")

    # group by true→pred
    from collections import defaultdict

    bucket = defaultdict(list)
    for r in mis:
        bucket[(r["true"], r["pred"])].append(r)

    for (t, p), rows in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1])):
        parts.append(
            f"<div class='card'><h2>True: {t} → Pred: {p} &nbsp; <span class='badge'>n={len(rows)}</span></h2>"
        )
        parts.append("<div class='grid'>")
        shown = 0
        for r in rows:
            if shown >= args.top_per_pair:
                break
            rel = ("../" + r["filepath"]).replace("\\", "/")
            parts.append(
                f"<div class='item'><a href='{rel}' target='_blank'><img src='{rel}'></a>"
            )
            meta = f"{r['pdf_base']}<br>vol={r['volunteer']} · seq={r['seq']}<br>margin={r['margin']:.3f}"
            parts.append(f"<div class='meta'>{meta}</div></div>")
            shown += 1
        parts.append("</div></div>")

    parts.append(HTML_TAIL)
    out_html.write_text("\n".join(parts), encoding="utf-8")
    print("Saved HTML:", out_html)


if __name__ == "__main__":
    main()
