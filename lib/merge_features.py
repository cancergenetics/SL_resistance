"""Shared library — merge per-feature outputs into withfeatures/dropna/dedup CSVs.

Reads feature_output/*.csv and writes (under data_dir):
  features_with_extracted.csv   — all 18 features merged onto the main schema
  features_dropna.csv           — drops rows missing co-expression, adds Class_Processed
  features_dropna_dedup.csv     — further dropped duplicates on all ML-relevant columns

The BiomarkerType NaN→0 imputation matches the training encoding (0 = not a known TSG).

Public entry point: merge_features(features_main_schema_csv, feature_output_dir, data_dir)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def select_highest_or_available(val1, val2):
    if pd.isna(val1) and pd.isna(val2):
        return np.nan
    if pd.isna(val1):
        return val2
    if pd.isna(val2):
        return val1
    return max(val1, val2, key=abs)


def merge_dual_target(col_target1_series: pd.Series, col_target2_series: pd.Series) -> list:
    return [select_highest_or_available(a, b) for a, b in zip(col_target1_series, col_target2_series)]


def merge_features(features_main_schema_csv: Path, feature_output_dir: Path,
                   data_dir: Path) -> None:
    features_main_schema_csv = Path(features_main_schema_csv)
    FEATURE_OUTPUT = Path(feature_output_dir)
    data_dir = Path(data_dir)

    FEATURES_WITH_EXTRACTED_CSV = data_dir / "features_with_extracted.csv"
    FEATURES_DROPNA_CSV         = data_dir / "features_dropna.csv"
    FEATURES_DROPNA_DEDUP_CSV   = data_dir / "features_dropna_dedup.csv"

    main = pd.read_csv(features_main_schema_csv)

    # ---- STRING score ----
    string_biomarker = pd.read_csv(FEATURE_OUTPUT / "string_score_query_biomarker.csv")
    main["StringInteractionWithBiomarker"] = (
        string_biomarker["StringInteractionWithBiomarker"] / 1000
    ).fillna(0)

    string_t1 = pd.read_csv(FEATURE_OUTPUT / "string_score_query_target1.csv")
    string_t2 = pd.read_csv(FEATURE_OUTPUT / "string_score_query_target2.csv")
    main["StringInteractionWithTarget"] = merge_dual_target(
        string_t1["StringInteractionWithTarget"], string_t2["StringInteractionWithTarget2"]
    )
    main["StringInteractionWithTarget"] = (main["StringInteractionWithTarget"] / 1000).fillna(0)

    # ---- GTEx co-expression ----
    gtex_biomarker = pd.read_csv(FEATURE_OUTPUT / "gtex_co_expression_biomarker_query.csv")
    main["CoexpressionWithBiomarker"] = gtex_biomarker["spearman_corr"]

    gtex_t1 = pd.read_csv(FEATURE_OUTPUT / "gtex_co_expression_target1_query.csv")
    main["CoexpressionWithTarget"] = gtex_t1["spearman_corr"]

    main["AvgExpression"] = gtex_biomarker["A2_mean_expr"]
    main["ExpressionVariance"] = gtex_biomarker["A2_expr_variance"]

    # ---- Co-essentiality ----
    coess_biomarker = pd.read_csv(FEATURE_OUTPUT / "coessentiality_biomarker_query.csv")
    main["CoessentialityWithBiomarker"] = coess_biomarker["Correlation"]

    coess_t1 = pd.read_csv(FEATURE_OUTPUT / "coessentiality_target1_query.csv")
    coess_t2 = pd.read_csv(FEATURE_OUTPUT / "coessentiality_target2_query.csv")
    main["CoessentialityWithTarget"] = merge_dual_target(coess_t1["Correlation"], coess_t2["Correlation"])

    main["EssentialityVariance"] = coess_biomarker["Biomarker_Essentiality_Variance"].fillna(0)
    main["EssentialityAverage"] = coess_biomarker["Biomarker_Essentiality_Average"].fillna(0)

    # ---- Shared PPI (BIOGRID FET) ----
    shared_bio_bm = pd.read_csv(FEATURE_OUTPUT / "fet_ppi_overlap_biomarker_query.csv")
    main["FET_SharedInteractors_Biomarker_BIOGRID"] = shared_bio_bm["fet_ppi_overlap"]

    shared_bio_t1 = pd.read_csv(FEATURE_OUTPUT / "fet_ppi_overlap_target1_query.csv")
    shared_bio_t2 = pd.read_csv(FEATURE_OUTPUT / "fet_ppi_overlap_target2_query.csv")
    main["FET_SharedInteractors_Target_BIOGRID"] = merge_dual_target(
        shared_bio_t1["fet_ppi_overlap"], shared_bio_t2["fet_ppi_overlap"]
    )

    # ---- Shared PPI (STRING FET) ----
    shared_str_bm = pd.read_csv(FEATURE_OUTPUT / "fet_ppi_overlap_biomarker_query_string.csv")
    main["FET_SharedInteractors_Biomarker_STRING"] = shared_str_bm["fet_ppi_overlap"]

    shared_str_t1 = pd.read_csv(FEATURE_OUTPUT / "fet_ppi_overlap_target1_query_string.csv")
    shared_str_t2 = pd.read_csv(FEATURE_OUTPUT / "fet_ppi_overlap_target2_query_string.csv")
    main["FET_SharedInteractors_Target_STRING"] = merge_dual_target(
        shared_str_t1["fet_ppi_overlap"], shared_str_t2["fet_ppi_overlap"]
    )

    # ---- BIOGRID physical binary ----
    biogrid_bm = pd.read_csv(FEATURE_OUTPUT / "biogrid_biomarker_query.csv")
    main["BIOGRIDPhysicalInteractionQueryBiomarker"] = biogrid_bm["BIOGRIDPhysicalInteractionQueryBiomarker"]

    biogrid_t = pd.read_csv(FEATURE_OUTPUT / "biogrid_target_query.csv")
    main["BIOGRIDPhysicalInteractionQueryTarget"] = biogrid_t["BIOGRIDPhysicalInteractionQueryTarget"]

    # ---- BiomarkerType (with Step-1 NaN→0 imputation) ----
    biomarker_type = pd.read_csv(FEATURE_OUTPUT / "biomarker_type.csv")
    main["BiomarkerType"] = biomarker_type["TSG_Label"].fillna(0).astype(int)

    # ---- Essentiality percentage (per query) ----
    ess_pct = pd.read_csv(FEATURE_OUTPUT / "essentiality_percentage_per_gene.csv")
    main = main.merge(ess_pct, left_on="entrez_id_query", right_on="entrez_id", how="left")
    main = main.drop(columns=["entrez_id"])
    main["Essentiality_Percentage"] = main["Essentiality_Percentage"].fillna(0)

    FEATURES_WITH_EXTRACTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    main.to_csv(FEATURES_WITH_EXTRACTED_CSV, index=False)
    print(f"wrote {FEATURES_WITH_EXTRACTED_CSV}  shape={main.shape}")

    # ---- Dropna on co-expression, add Class_Processed ----
    main_dropna = main.dropna(subset=["CoexpressionWithBiomarker", "CoexpressionWithTarget"]).reset_index(drop=True)
    main_dropna["Class_Processed"] = main_dropna["Class"].map({"Resistance": 1, "Non-Resistance": 0})
    main_dropna.to_csv(FEATURES_DROPNA_CSV, index=False)
    print(f"wrote {FEATURES_DROPNA_CSV}  shape={main_dropna.shape}")

    # ---- Dedup on all ML-relevant columns ----
    dedup_subset = [
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
        "BIOGRIDPhysicalInteractionQueryBiomarker", "BIOGRIDPhysicalInteractionQueryTarget",
        "ExpressionVariance", "EssentialityVariance", "EssentialityAverage",
        "Class_Processed",
    ]
    main_unique = main_dropna.drop_duplicates(subset=dedup_subset)
    main_unique.to_csv(FEATURES_DROPNA_DEDUP_CSV, index=False)
    print(f"wrote {FEATURES_DROPNA_DEDUP_CSV}  shape={main_unique.shape}")
