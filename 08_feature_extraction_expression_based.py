#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd

# Shared GTEx co-expression computation core.
from lib.feature_extraction_expression import (
    load_gtex_subset_for_genes,
    CalculateExprCorrelation_Fast,
    remove_duplicates_like_notebook,
    GTEX_CHUNKSIZE,
)


# ==========================================================
# A) CONFIG  (paths/names EXACTLY as your notebook)
# ==========================================================
MAIN_DATASET_CSV = "./input_data/3_ML_outputs/datasets/PredictingSLResistanceFeatures_main.csv"
GTEX_GCT_GZ = "./input_data/GTEx/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"

FEATURE_OUT_DIR = "./input_data/3_ML_outputs/feature_output"
OUT_BIOMARKER = os.path.join(FEATURE_OUT_DIR, "gtex_co_expression_biomarker_query.csv")
OUT_TARGET1   = os.path.join(FEATURE_OUT_DIR, "gtex_co_expression_target1_query.csv")
OUT_TARGET2   = os.path.join(FEATURE_OUT_DIR, "gtex_co_expression_target2_query.csv")
OUT_TARGET3   = os.path.join(FEATURE_OUT_DIR, "gtex_co_expression_target3_query.csv")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_main_dataset(path: str = MAIN_DATASET_CSV) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def build_pairs(main_df: pd.DataFrame, a_col: str, b_col: str) -> pd.DataFrame:
    return main_df[[a_col, b_col]].rename(columns={a_col: "A1_ensembl", b_col: "A2_ensembl"})


# ==========================================================
# ORCHESTRATION (computation lives in lib.feature_extraction_expression)
# ==========================================================
def run_one_coexpression(main_df: pd.DataFrame, a_col: str, b_col: str, out_csv: str) -> pd.DataFrame:
    pairs = build_pairs(main_df, a_col, b_col)
    genes = pd.unique(pairs[["A1_ensembl", "A2_ensembl"]].values.ravel())

    df_genes = load_gtex_subset_for_genes(GTEX_GCT_GZ, genes, chunksize=GTEX_CHUNKSIZE)
    df_genes_reset = df_genes.reset_index().rename(columns={"index": "ensembl_id"})
    expr_data = remove_duplicates_like_notebook(df_genes_reset)

    out = CalculateExprCorrelation_Fast(pairs, expr_data)
    out.to_csv(out_csv, index=False)
    return out


def main() -> None:
    ensure_dir(FEATURE_OUT_DIR)

    main_df = load_main_dataset(MAIN_DATASET_CSV)

    run_one_coexpression(main_df, "ensembl_gene_id_biomarker", "ensembl_gene_id_query", OUT_BIOMARKER)
    run_one_coexpression(main_df, "ensembl_gene_id_target1",   "ensembl_gene_id_query", OUT_TARGET1)
    run_one_coexpression(main_df, "ensembl_gene_id_target2",   "ensembl_gene_id_query", OUT_TARGET2)
    run_one_coexpression(main_df, "ensembl_gene_id_target3",   "ensembl_gene_id_query", OUT_TARGET3)

    print("[OK] Wrote:")
    print(" -", OUT_BIOMARKER)
    print(" -", OUT_TARGET1)
    print(" -", OUT_TARGET2)
    print(" -", OUT_TARGET3)


main()
