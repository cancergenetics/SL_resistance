#!/usr/bin/env python
# coding: utf-8


import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Shared GTEx co-expression computation core.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from lib.feature_extraction_expression import (
    load_gtex_subset_for_genes,
    CalculateExprCorrelation_Fast,
    remove_duplicates_like_notebook,
)

# =========================
# Configuration
# =========================
CONFIG = {
    "dataset_dir": "datasets",
    "feature_dir": "feature_output",
    "gtex_gct_gz": "../input_data/GTEx/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz",
    "pairs_xlsx": "biomarker_target_genes.xlsx",
    "gtex_chunksize": 1000,
}


# =========================
# Clinical-specific I/O orchestration
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


def build_pairs(combined: pd.DataFrame, partner: str) -> pd.DataFrame:
    assert partner in ("biomarker", "target1")
    col = f"ensembl_gene_id_{partner}"
    out = combined[["SL_Pair", col, "ensembl_gene_id_query"]].copy()
    out = out.rename(columns={col: "A1_ensembl", "ensembl_gene_id_query": "A2_ensembl"})
    return out.reset_index(drop=True)


def run_query_expression(
    combined: pd.DataFrame,
    partner: str,
    expr_data: pd.DataFrame,
    feature_dir: str,
) -> None:
    pairs = build_pairs(combined, partner)
    sl_pair_series = pairs["SL_Pair"].reset_index(drop=True)
    out = CalculateExprCorrelation_Fast(
        pairs.drop(columns=["SL_Pair"]),
        expr_data,
    ).reset_index(drop=True)
    out["SL_Pair"] = sl_pair_series

    for sl_pair, sub in out.groupby("SL_Pair", sort=False):
        name = f"gtex_co_expression_{partner}_query_{sl_pair}.csv"
        path = os.path.join(feature_dir, name)
        sub.drop(columns=["SL_Pair"]).to_csv(path, index=False)
        print(f"[OK] {path}  rows={len(sub):,}")


# =========================
# Run — single GTEx load over the union, per-partner correlation, per-pair split-write
# =========================
ensure_dir(CONFIG["feature_dir"])

pairs_df = pd.read_excel(CONFIG["pairs_xlsx"])
combined = load_combined_clinical_main(CONFIG["dataset_dir"], pairs_df)
print(f"[info] Combined clinical main: rows={len(combined):,}")

ensembl_union = pd.unique(
    combined[["ensembl_gene_id_biomarker", "ensembl_gene_id_target1", "ensembl_gene_id_query"]]
    .values.ravel()
)
print(f"[info] Unique ensembl IDs needed: {len(ensembl_union):,}")

df_genes = load_gtex_subset_for_genes(CONFIG["gtex_gct_gz"], ensembl_union, CONFIG["gtex_chunksize"])
df_genes_reset = df_genes.reset_index().rename(columns={"index": "ensembl_id"})
expr_data = remove_duplicates_like_notebook(df_genes_reset)
print(f"[info] GTEx expression matrix: {expr_data.shape}")

run_query_expression(combined, "biomarker", expr_data, CONFIG["feature_dir"])
run_query_expression(combined, "target1", expr_data, CONFIG["feature_dir"])
