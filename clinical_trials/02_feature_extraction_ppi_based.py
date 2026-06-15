#!/usr/bin/env python
# coding: utf-8


import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Shared PPI computation core (strict 3-tier partner mapping via lib.hgnc_lookup).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from lib.feature_extraction_ppi import (
    process_biogrid_ppi_data,
    compute_ppi_summary_for_pairs,
    load_and_map_string_all,
    build_string_unique_edges,
    build_string_score_dict,
    load_and_map_biogrid_physical,
    build_biogrid_exists_dict,
)

# =========================
# Configuration
# =========================
CONFIG = {
    "dataset_dir": "datasets",
    "feature_dir": "feature_output",
    "pairs_xlsx": "biomarker_target_genes.xlsx",
    "hgnc_file": "../input_data/HGNC/hgnc_complete_set.txt",
    "biogrid_all_tsv": "../input_data/BIOGRID/BIOGRID-ALL-4.4.241.tab3.txt",
    "string_links": "../input_data/STRING/9606.protein.links.detailed.v12.0.txt",
    "string_info": "../input_data/STRING/9606.protein.info.v12.0.txt",
    "cancer_gene_census_csv": "../input_data/Cancer_Gene_Census/Cancer_gene_census_data.csv",
    "string_score_threshold": 400,
}


# =========================
# Clinical-specific I/O orchestration
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_combined_clinical_main(dataset_dir: str, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Load each per-pair main CSV produced by notebook 1 and concatenate into one frame."""
    frames: List[pd.DataFrame] = []
    for _, row in pairs_df.iterrows():
        sl_pair = f"{row['Biomarker']}_{row['Target']}"
        path = os.path.join(dataset_dir, f"PredictingSLResistanceFeatures_{sl_pair}_main.csv")
        frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def build_entrez_pairs(combined: pd.DataFrame, partner: str) -> pd.DataFrame:
    """Extract (partner, query) entrez pairs from the combined clinical main,
    keeping SL_Pair for the split-write step. Used by BIOGRID-ALL FET and STRING FET."""
    assert partner in ("biomarker", "target1")
    col = f"entrez_id_{partner}"
    out = combined[["SL_Pair", col, "entrez_id_query"]].copy()
    out = out.rename(columns={col: "A1_entrez", "entrez_id_query": "A2_entrez"})
    out["A1_entrez"] = pd.to_numeric(out["A1_entrez"], errors="coerce")
    out["A2_entrez"] = pd.to_numeric(out["A2_entrez"], errors="coerce")
    out = out.dropna(subset=["A1_entrez", "A2_entrez"]).reset_index(drop=True)
    out["A1_entrez"] = out["A1_entrez"].astype(int)
    out["A2_entrez"] = out["A2_entrez"].astype(int)
    return out


def run_biogrid_fet_per_partner(
    biogrid_unique: pd.DataFrame,
    combined: pd.DataFrame,
    partner: str,
    feature_dir: str,
) -> None:
    pairs_df = build_entrez_pairs(combined, partner)
    sl_pair_series = pairs_df["SL_Pair"].reset_index(drop=True)
    fet_df = compute_ppi_summary_for_pairs(biogrid_unique, pairs_df.drop(columns=["SL_Pair"]))
    fet_df = fet_df.reset_index(drop=True)
    fet_df["SL_Pair"] = sl_pair_series
    for sl_pair, sub in fet_df.groupby("SL_Pair", sort=False):
        out_path = os.path.join(feature_dir, f"fet_ppi_overlap_{partner}_query_{sl_pair}.csv")
        sub.drop(columns=["SL_Pair"]).to_csv(out_path, index=False)
        print(f"[OK] {out_path}  rows={len(sub):,}")


def run_string_fet(
    string_unique_edges: pd.DataFrame,
    combined: pd.DataFrame,
    partner: str,
    feature_dir: str,
) -> None:
    pairs = build_entrez_pairs(combined, partner)
    sl_pair_series = pairs["SL_Pair"].reset_index(drop=True)
    fet_df = compute_ppi_summary_for_pairs(string_unique_edges, pairs.drop(columns=["SL_Pair"]))
    fet_df = fet_df.reset_index(drop=True)
    fet_df["SL_Pair"] = sl_pair_series
    for sl_pair, sub in fet_df.groupby("SL_Pair", sort=False):
        path = os.path.join(feature_dir, f"fet_ppi_overlap_{partner}_query_string_{sl_pair}.csv")
        sub.drop(columns=["SL_Pair"]).to_csv(path, index=False)
        print(f"[OK] {path}  rows={len(sub):,}")


def run_string_scores(
    combined: pd.DataFrame,
    score_dict: Dict[Tuple[object, object], float],
    feature_dir: str,
) -> None:
    """STRING combined_score stays raw (0-1000); the /1000 rescale happens in
    5_feature_merge to match main notebook 10."""
    def lookup_score(a, b):
        return score_dict.get((a, b)) or score_dict.get((b, a))

    for partner, out_col in [
        ("biomarker", "StringInteractionWithBiomarker"),
        ("target1", "StringInteractionWithTarget"),
    ]:
        ensembl_col = f"ensembl_gene_id_{partner}"
        sub = combined[["SL_Pair", "ensembl_gene_id_query", ensembl_col]].copy()
        sub[out_col] = sub.apply(
            lambda r: lookup_score(r["ensembl_gene_id_query"], r[ensembl_col]),
            axis=1,
        )
        for sl_pair, grp in sub.groupby("SL_Pair", sort=False):
            path = os.path.join(feature_dir, f"string_score_query_{partner}_{sl_pair}.csv")
            grp[["ensembl_gene_id_query", ensembl_col, out_col]].to_csv(path, index=False)
            print(f"[OK] {path}  rows={len(grp):,}")


def run_biogrid_physical_existence(
    combined: pd.DataFrame,
    biogrid_dict: Dict[Tuple[object, object], int],
    feature_dir: str,
) -> None:
    def exists(a, b):
        return 1 if (a, b) in biogrid_dict or (b, a) in biogrid_dict else 0

    for partner, out_col in [
        ("biomarker", "BIOGRIDPhysicalInteractionQueryBiomarker"),
        ("target1", "BIOGRIDPhysicalInteractionQueryTarget1"),
    ]:
        ensembl_col = f"ensembl_gene_id_{partner}"
        sub = combined[["SL_Pair", "ensembl_gene_id_query", ensembl_col]].copy()
        sub[out_col] = sub.apply(lambda r: exists(r["ensembl_gene_id_query"], r[ensembl_col]), axis=1)
        for sl_pair, grp in sub.groupby("SL_Pair", sort=False):
            path = os.path.join(feature_dir, f"biogrid_{partner}_query_{sl_pair}.csv")
            grp[["ensembl_gene_id_query", ensembl_col, out_col]].to_csv(path, index=False)
            print(f"[OK] {path}  rows={len(grp):,}")


# =========================
# Cancer Gene Census TSG / oncogene classification (clinical: + MTAP override)
# =========================
def classify_biomarker_tsg_oncogene(combined: pd.DataFrame, census_df: pd.DataFrame) -> pd.DataFrame:
    tmp = combined[["SL_Pair", "Biomarker"]].copy()
    tmp["Biomarker"] = tmp["Biomarker"].astype(str)

    census = census_df.copy()
    census["Gene Symbol"] = census["Gene Symbol"].astype(str)
    role_mapping = census.set_index("Gene Symbol")["Role in Cancer"].to_dict()

    # Manual loss-of-function annotation for biomarkers absent from CGC.
    # CGC lists the classic TSG/oncogene drivers; some SL biomarkers are not in it.
    # BiomarkerType encodes loss(1)/activating(0): MTAP's alteration is a homozygous
    # deletion (loss-of-function) -> 1.
    LOSS_OF_FUNCTION_BIOMARKERS = {"MTAP"}

    def classify_gene(gene_symbol):
        role = role_mapping.get(gene_symbol, None)
        if role and not pd.isna(role):
            if "TSG" in role:
                return 1
            elif "oncogene" in role:
                return 0
        if gene_symbol in LOSS_OF_FUNCTION_BIOMARKERS:
            return 1
        return np.nan

    tmp["TSG_Label"] = tmp["Biomarker"].apply(classify_gene)
    return tmp


def run_biomarker_type(combined: pd.DataFrame, census_df: pd.DataFrame, feature_dir: str) -> None:
    labeled = classify_biomarker_tsg_oncogene(combined, census_df)
    for sl_pair, grp in labeled.groupby("SL_Pair", sort=False):
        path = os.path.join(feature_dir, f"biomarker_type_{sl_pair}.csv")
        grp[["Biomarker", "TSG_Label"]].to_csv(path, index=False)
        print(f"[OK] {path}  rows={len(grp):,}")


# =========================
# Run — every heavy resource loads once
# =========================
ensure_dir(CONFIG["feature_dir"])

pairs_df = pd.read_excel(CONFIG["pairs_xlsx"])
combined = load_combined_clinical_main(CONFIG["dataset_dir"], pairs_df)
print(f"[info] Combined clinical main: rows={len(combined):,}  pairs={combined['SL_Pair'].nunique()}")

# --- BIOGRID-ALL shared interactors FET ---
biogrid_all_raw = pd.read_csv(CONFIG["biogrid_all_tsv"], sep="\t", low_memory=False)
biogrid_unique = process_biogrid_ppi_data(biogrid_all_raw)
print(f"[info] BIOGRID-ALL unique physical edges: {len(biogrid_unique):,}")

run_biogrid_fet_per_partner(biogrid_unique, combined, "biomarker", CONFIG["feature_dir"])
run_biogrid_fet_per_partner(biogrid_unique, combined, "target1", CONFIG["feature_dir"])

# --- STRING shared interactors FET + raw scores (3-tier partner mapping via lib) ---
string_all = load_and_map_string_all(CONFIG["string_links"], CONFIG["string_info"], CONFIG["hgnc_file"])
print(f"[info] STRING edges (all): {len(string_all):,}")

string_score_dict = build_string_score_dict(string_all)

string_filtered = string_all[string_all["combined_score"] >= CONFIG["string_score_threshold"]]
string_unique_edges = build_string_unique_edges(string_filtered)
print(f"[info] STRING unique edges >= {CONFIG['string_score_threshold']}: {len(string_unique_edges):,}")

run_string_fet(string_unique_edges, combined, "biomarker", CONFIG["feature_dir"])
run_string_fet(string_unique_edges, combined, "target1", CONFIG["feature_dir"])
run_string_scores(combined, string_score_dict, CONFIG["feature_dir"])

# --- BIOGRID Physical binary existence (3-tier partner mapping via lib) ---
biogrid_physical = load_and_map_biogrid_physical(CONFIG["biogrid_all_tsv"], CONFIG["hgnc_file"])
biogrid_exists_dict = build_biogrid_exists_dict(biogrid_physical)
run_biogrid_physical_existence(combined, biogrid_exists_dict, CONFIG["feature_dir"])

# --- Cancer Gene Census TSG / oncogene labels ---
census_df = pd.read_csv(CONFIG["cancer_gene_census_csv"], low_memory=False)
run_biomarker_type(combined, census_df, CONFIG["feature_dir"])

print("[OK] 2_feature_extraction_ppi_based complete")
