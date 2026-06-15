#!/usr/bin/env python
# coding: utf-8


import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Shared multi-target combine helper (single source of truth). Clinical pairs have
# only Target1 so this collapses to Target1 verbatim — imported for parity with 10.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from lib.merge_features import select_highest_or_available

# =========================
# Configuration
# =========================
CONFIG = {
    "pairs_xlsx": "biomarker_target_genes.xlsx",
    "dataset_dir": "datasets",
    "feature_dir": "feature_output",
    "string_rescale": 1000.0,
}

FEATURE_COLUMNS: List[str] = [
    "StringInteractionWithBiomarker",
    "StringInteractionWithTarget",
    "CoexpressionWithBiomarker",
    "CoexpressionWithTarget",
    "AvgExpression",
    "CoessentialityWithBiomarker",
    "CoessentialityWithTarget",
    "FET_SharedInteractors_Biomarker_BIOGRID",
    "FET_SharedInteractors_Target_BIOGRID",
    "FET_SharedInteractors_Biomarker_STRING",
    "FET_SharedInteractors_Target_STRING",
    "BIOGRIDPhysicalInteractionQueryBiomarker",
    "BIOGRIDPhysicalInteractionQueryTarget",
    "ExpressionVariance",
    "EssentialityVariance",
    "EssentialityAverage",
    "Essentiality_Percentage",
]



# =========================
# Utilities
# =========================
def read_feature_csv(feature_dir: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(feature_dir, filename), low_memory=False)


def load_essentiality_percentage(feature_dir: str) -> pd.DataFrame:
    """Pair-independent lookup: entrez_id -> Essentiality_Percentage."""
    return read_feature_csv(feature_dir, "essentiality_percentage_per_gene.csv")



# =========================
# Per-pair feature merge
# =========================
def merge_one_pair(
    sl_pair: str,
    dataset_dir: str,
    feature_dir: str,
    essentiality_percentage: pd.DataFrame,
    string_rescale: float,
) -> pd.DataFrame:
    """Attach all features to the per-pair main dataset and return the
    merged withfeatures frame. Clinical pairs have no Target2, so Target2
    feature slots stay NaN and ``select_highest_or_available`` collapses
    to Target1 verbatim."""
    main_path = os.path.join(dataset_dir, f"PredictingSLResistanceFeatures_{sl_pair}_main.csv")
    main = pd.read_csv(main_path, low_memory=False)

    # --- STRING scores (rescaled /1000 to match main notebook 10)
    string_bio = read_feature_csv(feature_dir, f"string_score_query_biomarker_{sl_pair}.csv")
    string_t1 = read_feature_csv(feature_dir, f"string_score_query_target1_{sl_pair}.csv")
    main["StringInteractionWithBiomarker"] = (
        string_bio["StringInteractionWithBiomarker"].astype(float) / string_rescale
    ).fillna(0)
    main["StringInteractionWithTarget"] = (
        string_t1["StringInteractionWithTarget"].astype(float) / string_rescale
    ).fillna(0)

    # --- GTEx expression correlation + query avg / variance
    gtex_bio = read_feature_csv(feature_dir, f"gtex_co_expression_biomarker_query_{sl_pair}.csv")
    gtex_t1 = read_feature_csv(feature_dir, f"gtex_co_expression_target1_query_{sl_pair}.csv")
    main["CoexpressionWithBiomarker"] = gtex_bio["spearman_corr"]
    main["CoexpressionWithTarget"] = gtex_t1["spearman_corr"]
    main["AvgExpression"] = gtex_bio["A2_mean_expr"]
    main["ExpressionVariance"] = gtex_bio["A2_expr_variance"]

    # --- Coessentiality (DepMap)
    coess_bio = read_feature_csv(feature_dir, f"coessentiality_biomarker_query_{sl_pair}.csv")
    coess_t1 = read_feature_csv(feature_dir, f"coessentiality_target1_query_{sl_pair}.csv")
    main["CoessentialityWithBiomarker"] = coess_bio["Correlation"]
    main["CoessentialityWithTarget"] = coess_t1["Correlation"]
    main["EssentialityVariance"] = coess_bio["Biomarker_Essentiality_Variance"].fillna(0)
    main["EssentialityAverage"] = coess_bio["Biomarker_Essentiality_Average"].fillna(0)

    # --- Shared interactors FET (BIOGRID)
    fet_bio_bg = read_feature_csv(feature_dir, f"fet_ppi_overlap_biomarker_query_{sl_pair}.csv")
    fet_t1_bg = read_feature_csv(feature_dir, f"fet_ppi_overlap_target1_query_{sl_pair}.csv")
    main["FET_SharedInteractors_Biomarker_BIOGRID"] = fet_bio_bg["fet_ppi_overlap"]
    main["FET_SharedInteractors_Target_BIOGRID"] = fet_t1_bg["fet_ppi_overlap"]

    # --- Shared interactors FET (STRING)
    fet_bio_str = read_feature_csv(feature_dir, f"fet_ppi_overlap_biomarker_query_string_{sl_pair}.csv")
    fet_t1_str = read_feature_csv(feature_dir, f"fet_ppi_overlap_target1_query_string_{sl_pair}.csv")
    main["FET_SharedInteractors_Biomarker_STRING"] = fet_bio_str["fet_ppi_overlap"]
    main["FET_SharedInteractors_Target_STRING"] = fet_t1_str["fet_ppi_overlap"]

    # --- BIOGRID-MV physical binary interaction
    bg_bio = read_feature_csv(feature_dir, f"biogrid_biomarker_query_{sl_pair}.csv")
    bg_t1 = read_feature_csv(feature_dir, f"biogrid_target1_query_{sl_pair}.csv")
    main["BIOGRIDPhysicalInteractionQueryBiomarker"] = bg_bio["BIOGRIDPhysicalInteractionQueryBiomarker"]
    target_col = (
        "BIOGRIDPhysicalInteractionQueryTarget"
        if "BIOGRIDPhysicalInteractionQueryTarget" in bg_t1.columns
        else "BIOGRIDPhysicalInteractionQueryTarget1"
    )
    main["BIOGRIDPhysicalInteractionQueryTarget"] = bg_t1[target_col]

    # --- Biomarker loss(1) / activating(0) label
    # Resolved at the source in 02 (classify_gene, incl. the MTAP loss-of-function
    # override), so biomarker_type_*.csv is already 0/1 with no NaN.
    btype = read_feature_csv(feature_dir, f"biomarker_type_{sl_pair}.csv")
    main["BiomarkerType"] = btype["TSG_Label"].astype(int)

    # --- Essentiality percentage (pair-independent lookup on entrez_id_query)
    main = main.merge(
        essentiality_percentage,
        left_on="entrez_id_query",
        right_on="entrez_id",
        how="left",
    ).drop(columns=["entrez_id"])
    main["Essentiality_Percentage"] = main["Essentiality_Percentage"].fillna(0)

    # Drop query genes missing GTEx expression — matches main notebook 10.
    main = main.dropna(subset=["ensembl_gene_id_query"]).reset_index(drop=True)
    main = main.dropna(
        subset=["CoexpressionWithBiomarker", "CoexpressionWithTarget"]
    ).reset_index(drop=True)
    return main



# =========================
# Orchestration
# =========================
def build_all_withfeatures(
    pairs_df: pd.DataFrame,
    dataset_dir: str,
    feature_dir: str,
    string_rescale: float,
    feature_columns: List[str],
) -> Dict[str, pd.DataFrame]:
    essentiality_percentage = load_essentiality_percentage(feature_dir)
    merged: Dict[str, pd.DataFrame] = {}
    for _, row in pairs_df.iterrows():
        biomarker = str(row["Biomarker"]).strip()
        target1 = str(row["Target"]).strip()
        sl_pair = f"{biomarker}_{target1}"

        df = merge_one_pair(
            sl_pair=sl_pair,
            dataset_dir=dataset_dir,
            feature_dir=feature_dir,
            essentiality_percentage=essentiality_percentage,
            string_rescale=string_rescale,
        )

        missing_cols = [c for c in feature_columns if c not in df.columns]
        assert not missing_cols, f"{sl_pair}: missing feature columns {missing_cols}"
        nan_total = int(df[feature_columns].isna().sum().sum())
        assert nan_total == 0, (
            f"{sl_pair}: {nan_total} NaN values across feature columns — "
            "fillna upstream or investigate missing per-pair feature rows"
        )

        merged[sl_pair] = df
        print(f"[OK] merge {sl_pair}  rows={len(df):,}  cols={df.shape[1]}")
    return merged


def write_withfeatures(merged: Dict[str, pd.DataFrame], dataset_dir: str) -> None:
    for sl_pair, df in merged.items():
        out_path = os.path.join(
            dataset_dir,
            f"PredictingSLResistanceFeatures_{sl_pair}_main_withfeatures.csv",
        )
        df.to_csv(out_path, index=False)
        print(f"[WRITE] {out_path}")



# =========================
# Run
# =========================
pairs_df = pd.read_excel(CONFIG["pairs_xlsx"])
print(f"[info] Loaded {len(pairs_df)} biomarker-target pairs")

merged = build_all_withfeatures(
    pairs_df=pairs_df,
    dataset_dir=CONFIG["dataset_dir"],
    feature_dir=CONFIG["feature_dir"],
    string_rescale=CONFIG["string_rescale"],
    feature_columns=FEATURE_COLUMNS,
)
write_withfeatures(merged, CONFIG["dataset_dir"])

sample_pair = next(iter(merged))
merged[sample_pair].head()