# -*- coding: utf-8 -*-
# Streamlit Labeler for SBBrasil TrainSheets
# Run: python -m streamlit run unsupervised_kit/labeler_app.py -- --root "."

import sys, argparse
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


# ----------------- CLI (safe inside Streamlit) -----------------
def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument(
        "--root", default=".", help="Project root (has metadata/ and images)"
    )
    # Streamlit injects args; ignore unknown
    try:
        ns, _ = ap.parse_known_args()
    except SystemExit:
        ns = ap.parse_args([])
    return ns


# ----------------- Helpers -----------------
def load_image_safe(p: Path):
    try:
        return Image.open(p)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_sources(root: Path):
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)

    # priority 1: review_candidates.csv (feito para revisão)
    rcsv = meta / "review_candidates.csv"
    if rcsv.exists():
        df = pd.read_csv(rcsv)
        source = "review_candidates.csv"
    else:
        # priority 2: pred_view_results.csv (baseline)
        pcsv = meta / "pred_view_results.csv"
        if pcsv.exists():
            df = pd.read_csv(pcsv)
            # normaliza para o formato esperado
            cols_map = {
                "y_pred": "y_pred",
                "y_true": "y_true",
                "margin": "margin",
                "pdf_base": "pdf_base",
                "voluntario": "voluntario",
                "seq": "seq",
                "filepath": "filepath",
            }
            missing = [c for c in ["filepath", "y_pred"] if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in pred_view_results.csv: {missing}")
            if "y_true" not in df.columns:
                df["y_true"] = ""
            if "margin" not in df.columns:
                df["margin"] = np.nan
            if "pdf_base" not in df.columns:
                # tenta inferir do manifest depois
                df["pdf_base"] = ""
            if "voluntario" not in df.columns:
                df["voluntario"] = ""
            if "seq" not in df.columns:
                df["seq"] = ""
            df = df[
                [
                    "filepath",
                    "pdf_base",
                    "voluntario",
                    "seq",
                    "y_pred",
                    "margin",
                    "y_true",
                ]
            ]
            source = "pred_view_results.csv"
        else:
            raise FileNotFoundError(
                "Não encontrei nem metadata/review_candidates.csv nem metadata/pred_view_results.csv."
            )

    # se faltar pdf_base/vol/seq, tenta completar via manifest
    man = pd.read_csv(meta / "manifest.csv")
    man_small = man[["filepath", "pdf_base", "voluntario", "seq"]]
    df = df.merge(man_small, on="filepath", how="left", suffixes=("", "_man"))
    for c in ["pdf_base", "voluntario", "seq"]:
        if c in df.columns and f"{c}_man" in df.columns:
            df[c] = df[c].fillna(df[f"{c}_man"])
    df = df[
        ["filepath", "pdf_base", "voluntario", "seq", "y_pred", "margin", "y_true"]
    ].copy()

    # remove duplicados por filepath mantendo primeira ocorrência
    df = df.drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    return df, source


def load_existing_labels(meta_dir: Path):
    lab = meta_dir / "review_labels.csv"
    if lab.exists():
        try:
            df = pd.read_csv(lab)
            if "filepath" in df.columns and "y_true" in df.columns:
                return df[["filepath", "y_true"]].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=["filepath", "y_true"])


def save_labels(df_labels: pd.DataFrame, meta_dir: Path):
    out = meta_dir / "review_labels.csv"
    df_labels.to_csv(out, index=False)
    return out


def export_to_views(meta_dir: Path):
    """Mescla review_labels.csv -> views.csv (mantém existentes, atualiza novos)."""
    lab = meta_dir / "review_labels.csv"
    if not lab.exists():
        raise FileNotFoundError(
            "metadata/review_labels.csv não encontrado para exportar."
        )

    labs = pd.read_csv(lab)
    if "filepath" not in labs or "y_true" not in labs.columns:
        raise ValueError("review_labels.csv precisa ter colunas: filepath, y_true")

    labs = labs[labs["y_true"].astype(str).str.len() > 0].copy()
    labs = labs.rename(columns={"y_true": "view"})[["filepath", "view"]]

    views_path = meta_dir / "views.csv"
    if views_path.exists():
        views = pd.read_csv(views_path)
        views = views[~views["filepath"].isin(labs["filepath"])]
        views = pd.concat([views, labs], ignore_index=True)
    else:
        views = labs

    views.to_csv(views_path, index=False)
    return views_path


# ----------------- App -----------------
def main():
    st.set_page_config(page_title="SBBrasil Labeler", layout="wide")
    args = parse_args()
    ROOT = Path(args.root).resolve()
    META = ROOT / "metadata"

    st.title("SBBrasil TrainSheets — Interactive Labeler")
    st.caption(f"Root: `{ROOT}`")

    try:
        base_df, source = load_sources(ROOT)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # carrega labels anteriores (auto-merge)
    existing = load_existing_labels(META)
    if not existing.empty:
        base_df = base_df.merge(
            existing, on="filepath", how="left", suffixes=("", "_saved")
        )
        base_df["y_true"] = base_df["y_true_saved"].combine_first(base_df["y_true"])
        base_df = base_df.drop(
            columns=[c for c in base_df.columns if c.endswith("_saved")]
        )

    # filtros
    st.sidebar.header("Filters")
    pdfs = ["(all)"] + sorted(
        [p for p in base_df["pdf_base"].dropna().unique() if str(p).strip()]
    )
    sel_pdf = st.sidebar.selectbox("Dataset (pdf_base)", pdfs, index=0)
    preds = ["(all)"] + sorted(base_df["y_pred"].astype(str).unique().tolist())
    sel_pred = st.sidebar.selectbox("Predicted class", preds, index=0)
    sort_uncertain = st.sidebar.checkbox(
        "Sort by uncertainty (lowest margin first)", value=True
    )

    df = base_df.copy()
    if sel_pdf != "(all)":
        df = df[df["pdf_base"] == sel_pdf]
    if sel_pred != "(all)":
        df = df[df["y_pred"].astype(str) == sel_pred]

    if sort_uncertain and "margin" in df.columns:
        df = df.sort_values("margin", ascending=True)

    # progresso
    done = int(df["y_true"].astype(str).str.len().gt(0).sum())
    total = int(len(df))
    st.markdown(f"**Progress:** {done} / {total} labeled")

    # estado de navegação
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    max_idx = max(0, len(df) - 1)
    st.session_state.idx = int(np.clip(st.session_state.idx, 0, max_idx))

    # navegação topo
    c_prev, c_go, c_next = st.columns([1, 3, 1])
    with c_prev:
        if st.button("⬅ Prev", use_container_width=True) and st.session_state.idx > 0:
            st.session_state.idx -= 1
    with c_go:
        st.number_input(
            "Index",
            min_value=0,
            max_value=max_idx,
            value=st.session_state.idx,
            key="go_idx",
            step=1,
        )
        if st.session_state.go_idx != st.session_state.idx:
            st.session_state.idx = st.session_state.go_idx
    with c_next:
        if (
            st.button("Next ➡", use_container_width=True)
            and st.session_state.idx < max_idx
        ):
            st.session_state.idx += 1

    if len(df) == 0:
        st.info("Nada a mostrar com os filtros atuais.")
        st.stop()

    # item atual
    row = df.iloc[st.session_state.idx]
    rel = row["filepath"].replace("\\", "/")
    img_path = ROOT / rel
    img = load_image_safe(img_path)

    left, right = st.columns([1, 1])
    with left:
        if img:
            st.image(img, caption=rel, use_container_width=True)
        else:
            st.warning(f"(missing) {rel}")

    with right:
        st.subheader("Prediction")
        st.markdown(f"- **pdf_base:** {row['pdf_base']}")
        st.markdown(
            f"- **vol:** {row.get('voluntario','')}, **seq:** {row.get('seq','')}"
        )
        st.markdown(
            f"- **Pred:** `{row['y_pred']}`"
            + (
                f"  · **margin:** {row['margin']:.3f}"
                if pd.notna(row.get("margin", np.nan))
                else ""
            )
        )

        # rótulo atual (se houver)
        current = str(row.get("y_true") or "").strip()
        if current:
            st.success(f"Labeled: **{current}**")

        st.divider()
        st.markdown("### Validate / Fix label")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "👍 Correct (use predicted)", type="primary", use_container_width=True
            ):
                base_df.loc[base_df["filepath"] == row["filepath"], "y_true"] = row[
                    "y_pred"
                ]
        with col2:
            wrong = st.button("👎 Incorrect", use_container_width=True)

        choices = ["frontal", "occlusal", "lateral", "other"]
        if wrong:
            st.session_state.show_fix = True
        if "show_fix" in st.session_state and st.session_state.show_fix:
            sel = st.radio(
                "Choose the correct label", choices, horizontal=True, index=0
            )
            if st.button("Save corrected label", use_container_width=True):
                base_df.loc[base_df["filepath"] == row["filepath"], "y_true"] = sel
                st.session_state.show_fix = False

        # autosave
        if st.button("💾 Save progress (CSV)"):
            out = save_labels(base_df[["filepath", "y_true"]], META)
            st.success(f"Saved: {out}")

        st.divider()
        if st.button("Export to views.csv", type="secondary"):
            try:
                save_labels(base_df[["filepath", "y_true"]], META)
                vpath = export_to_views(META)
                st.success(f"views.csv atualizado em: {vpath}")
            except Exception as e:
                st.error(str(e))

    # rodapé: grid pequena de contexto
    st.divider()
    st.caption("Context (mini grid)")
    small = df.iloc[max(0, st.session_state.idx - 6) : st.session_state.idx + 6]
    gc = st.columns(6)
    for i, (_, r) in enumerate(small.iterrows()):
        p = ROOT / r["filepath"]
        im = load_image_safe(p)
        with gc[i % 6]:
            if im:
                st.image(im, use_container_width=True)
            st.caption(f"{Path(r['filepath']).name}\n({r['y_pred']})")


if __name__ == "__main__":
    main()
