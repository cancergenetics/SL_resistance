#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd

# Shared DepMap co-essentiality computation core.
from lib.feature_extraction_essentiality import (
    preprocess_depmap,
    compute_gene_effect_corr,
    coessentiality_variance,
    essentiality_average,
    essentiality_percentage_per_gene,
    build_coessentiality_biomarker_query,
    build_coessentiality_target_query,
    ESSENTIALITY_THRESHOLD,
)


# ==========================================================
# A) CONFIG (keep names/paths exactly)
# ==========================================================
DEPMAP_GENE_EFFECT = "./input_data/DepMap/CRISPRGeneEffect.csv"
MAIN_DATASET_CSV   = "./input_data/3_ML_outputs/datasets/PredictingSLResistanceFeatures_main.csv"

FEATURE_OUT_DIR = "./input_data/3_ML_outputs/feature_output"

OUT_COESS_BIOMARKER = os.path.join(FEATURE_OUT_DIR, "coessentiality_biomarker_query.csv")
OUT_COESS_TARGET1   = os.path.join(FEATURE_OUT_DIR, "coessentiality_target1_query.csv")
OUT_COESS_TARGET2   = os.path.join(FEATURE_OUT_DIR, "coessentiality_target2_query.csv")
OUT_COESS_TARGET3   = os.path.join(FEATURE_OUT_DIR, "coessentiality_target3_query.csv")

OUT_ESS_PERCENT = os.path.join(FEATURE_OUT_DIR, "essentiality_percentage_per_gene.csv")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ==========================================================
# RUN (computation lives in lib.feature_extraction_essentiality)
# ==========================================================
def main() -> None:
    ensure_dir(FEATURE_OUT_DIR)

    # 1) DepMap preprocessing + correlation
    depmap_raw = pd.read_csv(DEPMAP_GENE_EFFECT)
    depmap = preprocess_depmap(depmap_raw)
    depmap.columns = depmap.columns.astype(float)

    corr = compute_gene_effect_corr(depmap)
    corr.index = corr.index.astype(float)
    corr.columns = corr.columns.astype(float)

    var_s = coessentiality_variance(depmap)   # index: float entrez
    avg_s = essentiality_average(depmap)      # index: float entrez

    # 2) Main ML dataset
    main_df = pd.read_csv(MAIN_DATASET_CSV, low_memory=False)

    # 3) biomarker/query (corr + query variance + query mean*-1)
    build_coessentiality_biomarker_query(main_df, corr, var_s, avg_s).to_csv(OUT_COESS_BIOMARKER, index=False)

    # 4) target1/2/3 vs query (corr)
    build_coessentiality_target_query(main_df, corr, "entrez_id_target1").to_csv(OUT_COESS_TARGET1, index=False)
    build_coessentiality_target_query(main_df, corr, "entrez_id_target2").to_csv(OUT_COESS_TARGET2, index=False)
    build_coessentiality_target_query(main_df, corr, "entrez_id_target3").to_csv(OUT_COESS_TARGET3, index=False)

    # 5) Essentiality percentage per gene
    essentiality_percentage_per_gene(depmap, threshold=ESSENTIALITY_THRESHOLD).to_csv(OUT_ESS_PERCENT, index=False)

    print("[OK] Wrote:")
    for p in (OUT_COESS_BIOMARKER, OUT_COESS_TARGET1, OUT_COESS_TARGET2, OUT_COESS_TARGET3, OUT_ESS_PERCENT):
        print(" -", p)


main()
