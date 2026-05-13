#!/usr/bin/env python
# coding: utf-8
"""Step 10 — Merge per-feature outputs (PPI, expression, essentiality, BIOGRID,
TSG, Essentiality Percentage) onto the main schema. Writes:

  PredictingSLResistanceFeatures_main_withfeatures.csv             — all features merged
  PredictingSLResistanceFeatures_main_withfeatures_dropna_filtered.csv — drops rows missing co-expression
  PredictingSLResistanceFeatures_main_withfeatures_dropna_remove_duplicate.csv — final ML dataset
"""

import os

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_DIR  = "input_data/3_ML_outputs"
DATA_DIR   = f"{INPUT_DIR}/datasets"
FEAT_DIR   = f"{INPUT_DIR}/feature_output"

MAIN_CSV               = f"{DATA_DIR}/PredictingSLResistanceFeatures_main.csv"
OUT_WITH_FEATURES_CSV  = f"{DATA_DIR}/PredictingSLResistanceFeatures_main_withfeatures.csv"
OUT_DROPNA_CSV         = f"{DATA_DIR}/PredictingSLResistanceFeatures_main_withfeatures_dropna_filtered.csv"
OUT_DROPNA_DEDUP_CSV   = f"{DATA_DIR}/PredictingSLResistanceFeatures_main_withfeatures_dropna_remove_duplicate.csv"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


ensure_dir(DATA_DIR)


# ============================================================
# Helper
# ============================================================

def select_highest_or_available(val1, val2):
    """Pick the value with larger absolute magnitude; NaN if both NaN."""
    if pd.isna(val1) and pd.isna(val2):
        return np.nan
    if pd.isna(val1):
        return val2
    if pd.isna(val2):
        return val1
    return max(val1, val2, key=abs)


def merge_target1_target2(series1, series2):
    return [select_highest_or_available(a, b) for a, b in zip(series1, series2)]


# ============================================================
# Load main schema
# ============================================================

main = pd.read_csv(MAIN_CSV)


# ============================================================
# STRING SCORE
# ============================================================

string_interaction_biomarker = pd.read_csv(f"{FEAT_DIR}/string_score_query_biomarker.csv")
main["StringInteractionWithBiomarker"] = (
    string_interaction_biomarker["StringInteractionWithBiomarker"] / 1000
).fillna(0)

string_interaction_target1 = pd.read_csv(f"{FEAT_DIR}/string_score_query_target1.csv")
string_interaction_target2 = pd.read_csv(f"{FEAT_DIR}/string_score_query_target2.csv")
main["StringInteractionWithTarget"] = merge_target1_target2(
    string_interaction_target1["StringInteractionWithTarget"],
    string_interaction_target2["StringInteractionWithTarget2"],
)
main["StringInteractionWithTarget"] = (main["StringInteractionWithTarget"] / 1000).fillna(0)


# ============================================================
# Co-expression
# ============================================================

gtex_coexpression_biomarker = pd.read_csv(f"{FEAT_DIR}/gtex_co_expression_biomarker_query.csv")
main["CoexpressionWithBiomarker"] = gtex_coexpression_biomarker["spearman_corr"]

gtex_coexpression_target1 = pd.read_csv(f"{FEAT_DIR}/gtex_co_expression_target1_query.csv")
gtex_coexpression_target2 = pd.read_csv(f"{FEAT_DIR}/gtex_co_expression_target2_query.csv")
main["CoexpressionWithTarget"] = merge_target1_target2(
    gtex_coexpression_target1["spearman_corr"],
    gtex_coexpression_target2["spearman_corr"],
)

main["AvgExpression"]      = gtex_coexpression_biomarker["A2_mean_expr"]
main["ExpressionVariance"] = gtex_coexpression_biomarker["A2_expr_variance"]


# ============================================================
# Co-essentiality
# ============================================================

coessentiality_biomarker = pd.read_csv(f"{FEAT_DIR}/coessentiality_biomarker_query.csv")
main["CoessentialityWithBiomarker"] = coessentiality_biomarker["Correlation"]

coessentiality_target1 = pd.read_csv(f"{FEAT_DIR}/coessentiality_target1_query.csv")
coessentiality_target2 = pd.read_csv(f"{FEAT_DIR}/coessentiality_target2_query.csv")
main["CoessentialityWithTarget"] = merge_target1_target2(
    coessentiality_target1["Correlation"],
    coessentiality_target2["Correlation"],
)

main["EssentialityVariance"] = coessentiality_biomarker["Biomarker_Essentiality_Variance"].fillna(0)
main["EssentialityAverage"]  = coessentiality_biomarker["Biomarker_Essentiality_Average"].fillna(0)


# ============================================================
# Shared PPI — BIOGRID
# ============================================================

shared_ppi_biomarker_biogrid = pd.read_csv(f"{FEAT_DIR}/fet_ppi_overlap_biomarker_query.csv")
main["FET_SharedInteractors_Biomarker_BIOGRID"] = shared_ppi_biomarker_biogrid["fet_ppi_overlap"]

shared_ppi_target1_biogrid = pd.read_csv(f"{FEAT_DIR}/fet_ppi_overlap_target1_query.csv")
shared_ppi_target2_biogrid = pd.read_csv(f"{FEAT_DIR}/fet_ppi_overlap_target2_query.csv")
main["FET_SharedInteractors_Target_BIOGRID"] = merge_target1_target2(
    shared_ppi_target1_biogrid["fet_ppi_overlap"],
    shared_ppi_target2_biogrid["fet_ppi_overlap"],
)


# ============================================================
# Shared PPI — STRING
# ============================================================

shared_ppi_biomarker_string = pd.read_csv(f"{FEAT_DIR}/fet_ppi_overlap_biomarker_query_string.csv")
main["FET_SharedInteractors_Biomarker_STRING"] = shared_ppi_biomarker_string["fet_ppi_overlap"]

shared_ppi_target1_string = pd.read_csv(f"{FEAT_DIR}/fet_ppi_overlap_target1_query_string.csv")
shared_ppi_target2_string = pd.read_csv(f"{FEAT_DIR}/fet_ppi_overlap_target2_query_string.csv")
main["FET_SharedInteractors_Target_STRING"] = merge_target1_target2(
    shared_ppi_target1_string["fet_ppi_overlap"],
    shared_ppi_target2_string["fet_ppi_overlap"],
)


# ============================================================
# BIOGRID Physical binary interactions
# ============================================================

biogrid_biomarker_interaction = pd.read_csv(f"{FEAT_DIR}/biogrid_biomarker_query.csv")
main["BIOGRIDPhysicalInteractionQueryBiomarker"] = biogrid_biomarker_interaction["BIOGRIDPhysicalInteractionQueryBiomarker"]

biogrid_target_interaction = pd.read_csv(f"{FEAT_DIR}/biogrid_target_query.csv")
main["BIOGRIDPhysicalInteractionQueryTarget"] = biogrid_target_interaction["BIOGRIDPhysicalInteractionQueryTarget"]


# ============================================================
# TSG / oncogene label
# ============================================================

biomarker_type = pd.read_csv(f"{FEAT_DIR}/biomarker_type.csv")
main["BiomarkerType"] = biomarker_type["TSG_Label"]


# ============================================================
# Essentiality percentage
# ============================================================

essentiality_percentage = pd.read_csv(f"{FEAT_DIR}/essentiality_percentage_per_gene.csv")

main = main.merge(
    essentiality_percentage,
    left_on="entrez_id_query",
    right_on="entrez_id",
    how="left",
).drop(columns=["entrez_id"])
main["Essentiality_Percentage"] = main["Essentiality_Percentage"].fillna(0)


# ============================================================
# Save full merge
# ============================================================

main.to_csv(OUT_WITH_FEATURES_CSV, index=False)


# ============================================================
# Drop NaN co-expression rows + add Class_Processed
# ============================================================

main_dropna = main.dropna(
    subset=["CoexpressionWithBiomarker", "CoexpressionWithTarget"]
).reset_index(drop=True)

main_dropna["Class_Processed"] = main_dropna["Class"].map({
    "Resistance": 1,
    "Non-Resistance": 0,
})

main_dropna.to_csv(OUT_DROPNA_CSV, index=False)


# ============================================================
# Drop duplicates on ML-relevant columns
# ============================================================

DEDUP_SUBSET = [
    "SL_Pair", "Query", "Biomarker", "Target1", "Target2",
    "entrez_id_query", "hgnc_id_query", "ensembl_gene_id_query",
    "entrez_id_biomarker", "hgnc_id_biomarker", "ensembl_gene_id_biomarker",
    "entrez_id_target1", "hgnc_id_target1", "ensembl_gene_id_target1",
    "entrez_id_target2", "hgnc_id_target2", "ensembl_gene_id_target2",
    "StringInteractionWithBiomarker", "StringInteractionWithTarget",
    "CoexpressionWithBiomarker", "CoexpressionWithTarget", "AvgExpression",
    "CoessentialityWithBiomarker", "CoessentialityWithTarget",
    "FET_SharedInteractors_Biomarker_BIOGRID", "FET_SharedInteractors_Target_BIOGRID",
    "FET_SharedInteractors_Biomarker_STRING", "FET_SharedInteractors_Target_STRING",
    "BIOGRIDPhysicalInteractionQueryBiomarker",
    "BIOGRIDPhysicalInteractionQueryTarget",
    "ExpressionVariance", "EssentialityVariance", "EssentialityAverage",
    "Class_Processed",
]

main_unique = main_dropna.drop_duplicates(subset=DEDUP_SUBSET)
main_unique.to_csv(OUT_DROPNA_DEDUP_CSV, index=False)
