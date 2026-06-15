#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd

# Shared PPI computation core (strict 3-tier partner mapping via lib.hgnc_lookup).
from lib.feature_extraction_ppi import (
    classify_biomarker_tsg_oncogene,
    process_biogrid_ppi_data,
    compute_ppi_summary_for_pairs,
    load_and_map_string_all,
    build_string_unique_edges,
    build_string_score_dict,
    load_and_map_biogrid_physical,
    build_biogrid_exists_dict,
    STRING_SCORE_THRESHOLD,
)


# ==========================================================
# A) CONFIG (KEEP FILES EXACTLY)
# ==========================================================
HGNC_TSV = "input_data/HGNC/hgnc_complete_set.txt"

BIOGRID_ALL_TSV = "input_data/BIOGRID/BIOGRID-ALL-4.4.241.tab3.txt"

STRING_LINKS = "input_data/STRING/9606.protein.links.detailed.v12.0.txt"
STRING_INFO  = "input_data/STRING/9606.protein.info.v12.0.txt"
CANCER_GENE_CENSUS_CSV = "input_data/Cancer_Gene_Census/Cancer_gene_census_data.csv"

MAIN_DATASET_CSV = "input_data/3_ML_outputs/datasets/PredictingSLResistanceFeatures_main.csv"
FEATURE_OUT_DIR  = "input_data/3_ML_outputs/feature_output"

# Output files (exact)
OUT_FET_BIOGRID_BIOMARKER_QUERY = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_biomarker_query.csv")
OUT_FET_BIOGRID_TARGET1_QUERY   = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_target1_query.csv")
OUT_FET_BIOGRID_TARGET2_QUERY   = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_target2_query.csv")
OUT_FET_BIOGRID_TARGET3_QUERY   = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_target3_query.csv")

OUT_FET_STRING_BIOMARKER_QUERY  = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_biomarker_query_string.csv")
OUT_FET_STRING_TARGET1_QUERY    = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_target1_query_string.csv")
OUT_FET_STRING_TARGET2_QUERY    = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_target2_query_string.csv")
OUT_FET_STRING_TARGET3_QUERY    = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_target3_query_string.csv")

OUT_STRING_SCORE_BIOMARKER = os.path.join(FEATURE_OUT_DIR, "string_score_query_biomarker.csv")
OUT_STRING_SCORE_TARGET1   = os.path.join(FEATURE_OUT_DIR, "string_score_query_target1.csv")
OUT_STRING_SCORE_TARGET2   = os.path.join(FEATURE_OUT_DIR, "string_score_query_target2.csv")
OUT_STRING_SCORE_TARGET3   = os.path.join(FEATURE_OUT_DIR, "string_score_query_target3.csv")

OUT_BIOGRID_BIOMARKER_QUERY = os.path.join(FEATURE_OUT_DIR, "biogrid_biomarker_query.csv")
OUT_BIOGRID_TARGET_QUERY    = os.path.join(FEATURE_OUT_DIR, "biogrid_target_query.csv")

OUT_BIOMARKER_TSG_LABEL = os.path.join(FEATURE_OUT_DIR, "biomarker_type.csv")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ==========================================================
# RUN (orchestration only; computation lives in lib.feature_extraction_ppi)
# ==========================================================
def main() -> None:
    ensure_dir(FEATURE_OUT_DIR)

    main_csv = pd.read_csv(MAIN_DATASET_CSV, low_memory=False)

    # Gene pairs for shared interactors (biomarker/target1/2/3 vs query)
    biomarker_query = main_csv[["entrez_id_biomarker", "entrez_id_query"]].rename(
        columns={"entrez_id_biomarker": "A1_entrez", "entrez_id_query": "A2_entrez"}
    )
    target1_query = main_csv[["entrez_id_target1", "entrez_id_query"]].rename(
        columns={"entrez_id_target1": "A1_entrez", "entrez_id_query": "A2_entrez"}
    )
    target2_query = main_csv[["entrez_id_target2", "entrez_id_query"]].rename(
        columns={"entrez_id_target2": "A1_entrez", "entrez_id_query": "A2_entrez"}
    )
    target3_query = main_csv[["entrez_id_target3", "entrez_id_query"]].rename(
        columns={"entrez_id_target3": "A1_entrez", "entrez_id_query": "A2_entrez"}
    )

    # --------------------------
    # BIOGRID-ALL shared interactors (FET)
    # --------------------------
    biogrid_raw = pd.read_csv(BIOGRID_ALL_TSV, sep="\t", low_memory=False)
    biogrid_unique = process_biogrid_ppi_data(biogrid_raw)

    compute_ppi_summary_for_pairs(biogrid_unique, biomarker_query).to_csv(OUT_FET_BIOGRID_BIOMARKER_QUERY, index=False)
    compute_ppi_summary_for_pairs(biogrid_unique, target1_query).to_csv(OUT_FET_BIOGRID_TARGET1_QUERY, index=False)
    compute_ppi_summary_for_pairs(biogrid_unique, target2_query).to_csv(OUT_FET_BIOGRID_TARGET2_QUERY, index=False)
    compute_ppi_summary_for_pairs(biogrid_unique, target3_query).to_csv(OUT_FET_BIOGRID_TARGET3_QUERY, index=False)

    # --------------------------
    # STRING: FULL table (NO threshold) for SCORE
    # --------------------------
    string_all = load_and_map_string_all(STRING_LINKS, STRING_INFO, HGNC_TSV)
    string_score_dict = build_string_score_dict(string_all)

    # --------------------------
    # STRING: thresholded table (>=400) for SHARED INTERACTORS
    # --------------------------
    string_filtered = string_all[string_all["combined_score"] >= STRING_SCORE_THRESHOLD].sort_values(
        by="combined_score", ascending=False
    )
    string_unique = build_string_unique_edges(string_filtered)

    compute_ppi_summary_for_pairs(string_unique, biomarker_query).to_csv(OUT_FET_STRING_BIOMARKER_QUERY, index=False)
    compute_ppi_summary_for_pairs(string_unique, target1_query).to_csv(OUT_FET_STRING_TARGET1_QUERY, index=False)
    compute_ppi_summary_for_pairs(string_unique, target2_query).to_csv(OUT_FET_STRING_TARGET2_QUERY, index=False)
    compute_ppi_summary_for_pairs(string_unique, target3_query).to_csv(OUT_FET_STRING_TARGET3_QUERY, index=False)

    # --------------------------
    # STRING score outputs (NO threshold)
    # --------------------------
    main_csv2 = pd.read_csv(MAIN_DATASET_CSV, low_memory=False)

    def get_score_biomarker(row):
        pair1 = (row["ensembl_gene_id_query"], row["ensembl_gene_id_biomarker"])
        pair2 = (row["ensembl_gene_id_biomarker"], row["ensembl_gene_id_query"])
        return string_score_dict.get(pair1) or string_score_dict.get(pair2)

    def get_score_target1(row):
        pair1 = (row["ensembl_gene_id_query"], row["ensembl_gene_id_target1"])
        pair2 = (row["ensembl_gene_id_target1"], row["ensembl_gene_id_query"])
        return string_score_dict.get(pair1) or string_score_dict.get(pair2)

    def get_score_target2(row):
        target2 = row["ensembl_gene_id_target2"]
        if pd.isna(target2) or target2 == "":
            return 0
        pair1 = (row["ensembl_gene_id_query"], target2)
        pair2 = (target2, row["ensembl_gene_id_query"])
        return string_score_dict.get(pair1) or string_score_dict.get(pair2) or 0

    def get_score_target3(row):
        target3 = row["ensembl_gene_id_target3"]
        if pd.isna(target3) or target3 == "":
            return 0
        pair1 = (row["ensembl_gene_id_query"], target3)
        pair2 = (target3, row["ensembl_gene_id_query"])
        return string_score_dict.get(pair1) or string_score_dict.get(pair2) or 0

    main_csv2["StringInteractionWithBiomarker"] = main_csv2.apply(get_score_biomarker, axis=1)
    main_csv2[["ensembl_gene_id_query", "ensembl_gene_id_biomarker", "StringInteractionWithBiomarker"]].to_csv(
        OUT_STRING_SCORE_BIOMARKER, index=False
    )

    main_csv2["StringInteractionWithTarget"] = main_csv2.apply(get_score_target1, axis=1)
    main_csv2[["ensembl_gene_id_query", "ensembl_gene_id_target1", "StringInteractionWithTarget"]].to_csv(
        OUT_STRING_SCORE_TARGET1, index=False
    )

    main_csv2["StringInteractionWithTarget2"] = main_csv2.apply(get_score_target2, axis=1)
    main_csv2[["ensembl_gene_id_query", "ensembl_gene_id_target2", "StringInteractionWithTarget2"]].to_csv(
        OUT_STRING_SCORE_TARGET2, index=False
    )

    main_csv2["StringInteractionWithTarget3"] = main_csv2.apply(get_score_target3, axis=1)
    main_csv2[["ensembl_gene_id_query", "ensembl_gene_id_target3", "StringInteractionWithTarget3"]].to_csv(
        OUT_STRING_SCORE_TARGET3, index=False
    )

    # --------------------------
    # BIOGRID Physical existence outputs
    # --------------------------
    main_csv3 = pd.read_csv(MAIN_DATASET_CSV, low_memory=False)
    biogrid_physical = load_and_map_biogrid_physical(BIOGRID_ALL_TSV, HGNC_TSV)
    biogrid_dict = build_biogrid_exists_dict(biogrid_physical)

    def check_bio_biomarker(row):
        pair1 = (row["ensembl_gene_id_query"], row["ensembl_gene_id_biomarker"])
        pair2 = (row["ensembl_gene_id_biomarker"], row["ensembl_gene_id_query"])
        return 1 if pair1 in biogrid_dict or pair2 in biogrid_dict else 0

    def check_bio_target1(row):
        pair1 = (row["ensembl_gene_id_query"], row["ensembl_gene_id_target1"])
        pair2 = (row["ensembl_gene_id_target1"], row["ensembl_gene_id_query"])
        return 1 if pair1 in biogrid_dict or pair2 in biogrid_dict else 0

    def check_bio_target2(row):
        pair1 = (row["ensembl_gene_id_query"], row["ensembl_gene_id_target2"])
        pair2 = (row["ensembl_gene_id_target2"], row["ensembl_gene_id_query"])
        return 1 if pair1 in biogrid_dict or pair2 in biogrid_dict else 0

    def check_bio_target3(row):
        pair1 = (row["ensembl_gene_id_query"], row["ensembl_gene_id_target3"])
        pair2 = (row["ensembl_gene_id_target3"], row["ensembl_gene_id_query"])
        return 1 if pair1 in biogrid_dict or pair2 in biogrid_dict else 0

    main_csv3["BIOGRIDPhysicalInteractionQueryBiomarker"] = main_csv3.apply(check_bio_biomarker, axis=1)
    main_csv3["BIOGRIDPhysicalInteractionQueryTarget1"] = main_csv3.apply(check_bio_target1, axis=1)
    main_csv3["BIOGRIDPhysicalInteractionQueryTarget2"] = main_csv3.apply(check_bio_target2, axis=1)
    main_csv3["BIOGRIDPhysicalInteractionQueryTarget3"] = main_csv3.apply(check_bio_target3, axis=1)

    main_csv3["BIOGRIDPhysicalInteractionQueryTarget"] = (
        main_csv3["BIOGRIDPhysicalInteractionQueryTarget1"]
        | main_csv3["BIOGRIDPhysicalInteractionQueryTarget2"]
        | main_csv3["BIOGRIDPhysicalInteractionQueryTarget3"]
    ).astype(int)

    main_csv3[["ensembl_gene_id_query", "ensembl_gene_id_biomarker", "BIOGRIDPhysicalInteractionQueryBiomarker"]].to_csv(
        OUT_BIOGRID_BIOMARKER_QUERY, index=False
    )
    main_csv3[["ensembl_gene_id_query", "ensembl_gene_id_target1", "BIOGRIDPhysicalInteractionQueryTarget"]].to_csv(
        OUT_BIOGRID_TARGET_QUERY, index=False
    )

    # --------------------------
    # Biomarker TSG / oncogene label (Cancer Gene Census)
    # --------------------------
    try:
        cancer_gene_census = pd.read_csv(CANCER_GENE_CENSUS_CSV, low_memory=False)
        biomarker_type = classify_biomarker_tsg_oncogene(main_csv3, cancer_gene_census)
        biomarker_type.to_csv(OUT_BIOMARKER_TSG_LABEL, index=False)
        print("[OK] Biomarker TSG/oncogene labels written:", OUT_BIOMARKER_TSG_LABEL)
    except FileNotFoundError:
        print(f"[WARN] Cancer Gene Census file not found: {CANCER_GENE_CENSUS_CSV} (skipping TSG/oncogene labels)")
    except Exception as e:
        print(f"[WARN] Failed to compute Biomarker TSG/oncogene labels: {e}")

    print("[OK] Finished. Outputs written to:", FEATURE_OUT_DIR)


main()
