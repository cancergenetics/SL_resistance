#!/usr/bin/env python
# coding: utf-8


import os
import re
import numpy as np
import pandas as pd
from typing import List

# =========================
# Configuration
# =========================
CONFIG = {
    "dataset_dir": "datasets",
    "feature_dir": "feature_output",
    "depmap_crispr_csv": "../input_data/DepMap/CRISPRGeneEffect.csv",
    "pairs_xlsx": "biomarker_target_genes.xlsx",
    "essentiality_threshold": -0.6,
}



# =========================
# Utilities & I/O
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_combined_clinical_main(dataset_dir: str, pairs_df: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for _, row in pairs_df.iterrows():
        sl_pair = f"{row['Biomarker']}_{row['Target']}"
        path = os.path.join(dataset_dir, f"PredictingSLResistanceFeatures_{sl_pair}_main.csv")
        frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def extract_number(col_name: str) -> str:
    match = re.search(r"\((\d+)\)", str(col_name))
    return match.group(1) if match else str(col_name)


def preprocess_depmap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Unnamed: 0": "sample"}).copy()
    df = df.set_index(df["sample"])
    df = df.drop(columns=["sample"])
    df.columns = [extract_number(c) for c in df.columns]
    return df



# =========================
# Pure essentiality metrics (copy-adapted from 9_feature_extraction_essentiality_based.ipynb)
# =========================
def compute_gene_effect_corr(depmap_matrix: pd.DataFrame) -> pd.DataFrame:
    return depmap_matrix.corr(method="pearson")


def coessentiality_variance(depmap_matrix: pd.DataFrame) -> pd.Series:
    return depmap_matrix.var(axis=0)


def essentiality_average(depmap_matrix: pd.DataFrame) -> pd.Series:
    return depmap_matrix.mean(axis=0)


def coessentiality_lookup(corr: pd.DataFrame, a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return 0.0
    if (a in corr.index) and (b in corr.columns):
        return float(corr.loc[a, b])
    return 0.0


def essentiality_percentage_per_gene(depmap_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = {}
    for col in depmap_matrix.columns:
        if pd.api.types.is_numeric_dtype(depmap_matrix[col]):
            total = depmap_matrix[col].notna().sum()
            if total > 0:
                count = (depmap_matrix[col] < threshold).sum()
                result[col] = (count / total) * 100
            else:
                result[col] = None
    return pd.DataFrame(list(result.items()), columns=["entrez_id", "Essentiality_Percentage"])



# =========================
# Coessentiality builders for the combined clinical main
# =========================
def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def build_coessentiality_biomarker_query(
    combined: pd.DataFrame,
    corr: pd.DataFrame,
    var_s: pd.Series,
    avg_s: pd.Series,
) -> pd.DataFrame:
    out = combined[["SL_Pair", "entrez_id_query", "entrez_id_biomarker"]].copy()
    out["entrez_id_query"] = _to_float(out["entrez_id_query"])
    out["entrez_id_biomarker"] = _to_float(out["entrez_id_biomarker"])

    out["Correlation"] = out.apply(
        lambda r: coessentiality_lookup(corr, r["entrez_id_query"], r["entrez_id_biomarker"]),
        axis=1,
    )
    out["Biomarker_Essentiality_Variance"] = out["entrez_id_query"].map(
        lambda gid: var_s.get(gid, np.nan) if not pd.isna(gid) else np.nan
    )
    out["Biomarker_Essentiality_Average"] = out["entrez_id_query"].map(
        lambda gid: avg_s.get(gid, np.nan) if not pd.isna(gid) else np.nan
    )
    out["Biomarker_Essentiality_Average"] = out["Biomarker_Essentiality_Average"] * -1
    return out


def build_coessentiality_target1_query(combined: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    out = combined[["SL_Pair", "entrez_id_query", "entrez_id_target1"]].copy()
    out["entrez_id_query"] = _to_float(out["entrez_id_query"])
    out["entrez_id_target1"] = _to_float(out["entrez_id_target1"])
    out["Correlation"] = out.apply(
        lambda r: coessentiality_lookup(corr, r["entrez_id_query"], r["entrez_id_target1"]),
        axis=1,
    )
    return out


def split_and_write(out: pd.DataFrame, partner: str, feature_dir: str) -> None:
    for sl_pair, sub in out.groupby("SL_Pair", sort=False):
        name = f"coessentiality_{partner}_query_{sl_pair}.csv"
        path = os.path.join(feature_dir, name)
        sub.drop(columns=["SL_Pair"]).to_csv(path, index=False)
        print(f"[OK] {path}  rows={len(sub):,}")



# =========================
# Run
# =========================
ensure_dir(CONFIG["feature_dir"])

pairs_df = pd.read_excel(CONFIG["pairs_xlsx"])
combined = load_combined_clinical_main(CONFIG["dataset_dir"], pairs_df)
print(f"[info] Combined clinical main: rows={len(combined):,}")

depmap_raw = pd.read_csv(CONFIG["depmap_crispr_csv"])
depmap = preprocess_depmap(depmap_raw)
depmap.columns = depmap.columns.astype(float)
print(f"[info] DepMap matrix: {depmap.shape}")

corr = compute_gene_effect_corr(depmap)
corr.index = corr.index.astype(float)
corr.columns = corr.columns.astype(float)
var_s = coessentiality_variance(depmap)
avg_s = essentiality_average(depmap)

bio_out = build_coessentiality_biomarker_query(combined, corr, var_s, avg_s)
split_and_write(bio_out, "biomarker", CONFIG["feature_dir"])

t1_out = build_coessentiality_target1_query(combined, corr)
split_and_write(t1_out, "target1", CONFIG["feature_dir"])

ess_pct = essentiality_percentage_per_gene(depmap, threshold=CONFIG["essentiality_threshold"])
ess_pct_path = os.path.join(CONFIG["feature_dir"], "essentiality_percentage_per_gene.csv")
ess_pct.to_csv(ess_pct_path, index=False)
print(f"[OK] {ess_pct_path}  rows={len(ess_pct):,}")