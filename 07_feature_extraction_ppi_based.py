#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, Tuple, Optional


# ==========================================================
# A) CONFIG (KEEP FILES EXACTLY)
# ==========================================================
HGNC_TSV = "input_data/HGNC/hgnc_complete_set.txt"

BIOGRID_ALL_TSV = "input_data/BIOGRID/BIOGRID-ALL-4.4.241.tab3.txt"
BIOGRID_PHYSICAL_TSV = "input_data/BIOGRID/BIOGRID-ALL-4.4.241.tab3.txt"  # physical interactions filtered from BIOGRID-ALL

STRING_LINKS = "input_data/STRING/9606.protein.links.detailed.v12.0.txt"
STRING_INFO  = "input_data/STRING/9606.protein.info.v12.0.txt"
STRING_SCORE_THRESHOLD = 400   # ONLY for shared partners / STRING network
CANCER_GENE_CENSUS_CSV = "input_data/Cancer_Gene_Census/Cancer_gene_census_data.csv"

MAIN_DATASET_CSV = "input_data/3_ML_outputs/datasets/PredictingSLResistanceFeatures_main.csv"
FEATURE_OUT_DIR  = "input_data/3_ML_outputs/feature_output"

# Output files (exact)
OUT_FET_BIOGRID_BIOMARKER_QUERY = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_biomarker_query.csv")
OUT_FET_BIOGRID_TARGET1_QUERY   = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_target1_query.csv")
OUT_FET_BIOGRID_TARGET2_QUERY   = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_target2_query.csv")

OUT_FET_STRING_BIOMARKER_QUERY  = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_biomarker_query_string.csv")
OUT_FET_STRING_TARGET1_QUERY    = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_target1_query_string.csv")
OUT_FET_STRING_TARGET2_QUERY    = os.path.join(FEATURE_OUT_DIR, "fet_ppi_overlap_target2_query_string.csv")

OUT_STRING_SCORE_BIOMARKER = os.path.join(FEATURE_OUT_DIR, "string_score_query_biomarker.csv")
OUT_STRING_SCORE_TARGET1   = os.path.join(FEATURE_OUT_DIR, "string_score_query_target1.csv")
OUT_STRING_SCORE_TARGET2   = os.path.join(FEATURE_OUT_DIR, "string_score_query_target2.csv")

OUT_BIOGRID_BIOMARKER_QUERY = os.path.join(FEATURE_OUT_DIR, "biogrid_biomarker_query.csv")
OUT_BIOGRID_TARGET_QUERY    = os.path.join(FEATURE_OUT_DIR, "biogrid_target_query.csv")

OUT_BIOMARKER_TSG_LABEL = os.path.join(FEATURE_OUT_DIR, "biomarker_type.csv")

# ==========================================================
# B) SMALL HELPERS
# ==========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_symbol(s: Optional[str]) -> Optional[str]:
    return s.strip().upper() if isinstance(s, str) else None

def classify_biomarker_tsg_oncogene(main_df: pd.DataFrame, census_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates notebook logic:
      - map Cancer Gene Census 'Gene Symbol' -> 'Role in Cancer'
      - classify Biomarker:
          1 if 'TSG' in role
          0 if 'oncogene' in role
          NaN otherwise
    """
    # Notebook expects gene symbols (string)
    biomarker_col = "Biomarker"
    if biomarker_col not in main_df.columns:
        raise KeyError(f"'{biomarker_col}' column not found in MAIN dataset. Available columns: {list(main_df.columns)[:30]} ...")

    if "Gene Symbol" not in census_df.columns or "Role in Cancer" not in census_df.columns:
        raise KeyError("Cancer Gene Census file must contain columns: 'Gene Symbol' and 'Role in Cancer'.")

    tmp = main_df[[biomarker_col]].copy()
    tmp[biomarker_col] = tmp[biomarker_col].astype(str)

    census = census_df.copy()
    census["Gene Symbol"] = census["Gene Symbol"].astype(str)

    role_mapping = census.set_index("Gene Symbol")["Role in Cancer"].to_dict()

    def classify_gene(gene_symbol: str):
        role = role_mapping.get(gene_symbol, None)
        if role:
            if "TSG" in role:
                return 1
            elif "oncogene" in role:
                return 0
        return np.nan

    tmp["TSG_Label"] = tmp[biomarker_col].apply(classify_gene)
    return tmp

# ==========================================================
# C) BIOGRID-ALL (Entrez) SHARED INTERACTORS (FET)
# ==========================================================
def process_biogrid_ppi_data(data: pd.DataFrame) -> pd.DataFrame:
    hsapien = 9606
    biogrid_ppi = data[
        (data["Organism ID Interactor A"] == hsapien) &
        (data["Organism ID Interactor B"] == hsapien) &
        (data["Experimental System Type"] == "physical")
    ]
    biogrid_ppi = biogrid_ppi[["Entrez Gene Interactor A", "Entrez Gene Interactor B", "Experimental System"]]
    biogrid_ppi = biogrid_ppi.rename(columns={
        "Entrez Gene Interactor A": "A1_entrez",
        "Entrez Gene Interactor B": "A2_entrez",
        "Experimental System": "experimental_system"
    })
    biogrid_ppi = biogrid_ppi.drop_duplicates().reset_index(drop=True).astype({"A1_entrez": "int", "A2_entrez": "int"})
    biogrid_ppi = biogrid_ppi[biogrid_ppi["A1_entrez"] != biogrid_ppi["A2_entrez"]].reset_index(drop=True)

    biogrid_unique = pd.DataFrame(
        np.sort(biogrid_ppi[["A1_entrez", "A2_entrez"]], axis=1),
        columns=["A1_entrez", "A2_entrez"]
    ).drop_duplicates()

    return biogrid_unique.reset_index(drop=True)


def compute_ppi_summary_for_pairs(ppi: pd.DataFrame, gene_pairs: pd.DataFrame) -> pd.DataFrame:
    ppi_symmetric = pd.concat([ppi, ppi.rename(columns={"A1_entrez": "A2_entrez", "A2_entrez": "A1_entrez"})])
    ppi_symmetric = ppi_symmetric.reset_index(drop=True)
    assert ppi_symmetric.shape[0] == ppi.shape[0] * 2

    gene_ppi = pd.merge(gene_pairs[["A1_entrez", "A2_entrez"]], ppi_symmetric, how="left", indicator="interact")
    gene_ppi.interact = gene_ppi.interact == "both"
    assert gene_ppi.shape[0] == gene_pairs.shape[0]
    print("N. gene pairs that interact:", sum(gene_ppi.interact))

    ppi_per_gene = (
        ppi_symmetric.groupby("A1_entrez").agg({"A2_entrez": set}).reset_index()
        .rename(columns={"A1_entrez": "gene", "A2_entrez": "ppi"})
    )
    assert ppi_per_gene.shape[0] == pd.concat([ppi.A1_entrez, ppi.A2_entrez]).nunique()

    df = pd.merge(gene_ppi, ppi_per_gene.rename(columns={"gene": "A1_entrez", "ppi": "A1_ppi"}), how="left")
    df = pd.merge(df, ppi_per_gene.rename(columns={"gene": "A2_entrez", "ppi": "A2_ppi"}), how="left")
    assert df.shape[0] == gene_pairs.shape[0]
    print("N paralog pairs w/ 1+ interactor (A1 and/or A2):", df[(~df.A1_ppi.isna()) | (~df.A2_ppi.isna())].shape[0])

    df["A1_ppi"] = df["A1_ppi"].apply(lambda d: d if not pd.isnull(d) else set())
    df["A2_ppi"] = df["A2_ppi"].apply(lambda d: d if not pd.isnull(d) else set())

    df.A1_ppi = df.apply(lambda x: x.A1_ppi - {x.A2_entrez}, axis=1)
    df.A2_ppi = df.apply(lambda x: x.A2_ppi - {x.A1_entrez}, axis=1)

    df["n_A1_ppi"] = df.apply(lambda x: len(x.A1_ppi), axis=1)
    df["n_A2_ppi"] = df.apply(lambda x: len(x.A2_ppi), axis=1)
    df["shared_ppi"] = df.apply(lambda x: x.A1_ppi.intersection(x.A2_ppi), axis=1)
    df["n_total_ppi"] = df.apply(lambda x: len(x.A1_ppi.union(x.A2_ppi)), axis=1)
    df["n_shared_ppi"] = df.apply(lambda x: len(x.shared_ppi), axis=1)

    def calc_jaccard_index(x):
        if x.n_shared_ppi == 0:
            return 0
        return x.n_shared_ppi / ((x.n_A1_ppi + x.n_A2_ppi) - x.n_shared_ppi)

    df["shared_ppi_jaccard_idx"] = df.apply(calc_jaccard_index, axis=1)

    N = len(pd.concat([ppi.A1_entrez, ppi.A2_entrez]).unique())
    print("N genes involded in interactions:", N)
    assert ppi_per_gene.shape[0] == N

    def calc_fet_shared_ppi(x):
        ctab = pd.DataFrame(
            {"A2": [x.n_shared_ppi, x.n_A2_ppi - x.n_shared_ppi],
             "NA2": [x.n_A1_ppi - x.n_shared_ppi, N - x.n_total_ppi]},
            index=["A1", "NA1"]
        )
        (OR, pval) = stats.fisher_exact(ctab)
        if pval == 0:
            pval = np.nextafter(0, 1)
        log_pval = (-np.log10(pval)) if pval != 1 else 0
        log_pval = -log_pval if OR < 1 else log_pval
        return log_pval

    df["fet_ppi_overlap"] = df.apply(calc_fet_shared_ppi, axis=1)
    df = df.drop(columns=["A1_ppi", "A2_ppi", "n_A1_ppi", "n_A2_ppi"])
    return df


# ==========================================================
# D) STRING: mapping uses your original "first-win" HGNC lookup
#     - SCORE FEATURES: NO threshold
#     - SHARED PARTNERS: threshold 400
# ==========================================================
def build_hgnc_lookup_firstwin(hgnc_dict) -> Dict[str, Tuple[object, object, str]]:
    gene_lookup = {}
    for entry in hgnc_dict:
        hgnc_id = entry.get("hgnc_id", np.nan)
        ensembl_id = entry.get("ensembl_gene_id", np.nan)
        entrez_id = str(entry.get("entrez_id", "NA")) if pd.notnull(entry.get("entrez_id")) else "NA"

        syms = []
        s = normalize_symbol(entry.get("symbol"))
        if s:
            syms.append(s)

        if pd.notnull(entry.get("prev_symbol")):
            syms.extend(normalize_symbol(x) for x in str(entry["prev_symbol"]).split("|"))

        if pd.notnull(entry.get("alias_symbol")):
            syms.extend(normalize_symbol(x) for x in str(entry["alias_symbol"]).split("|"))

        for sym in syms:
            if sym and sym not in gene_lookup:
                gene_lookup[sym] = (hgnc_id, ensembl_id, entrez_id)

    return gene_lookup


def lookup_gene_info_firstwin(gene_lookup, gene_symbol: str) -> Tuple[object, object, str]:
    key = normalize_symbol(gene_symbol)
    return gene_lookup.get(key, (np.nan, np.nan, "NA"))


def load_and_map_string_all(hgnc_dict) -> pd.DataFrame:
    # NOTE: This is the FULL STRING (no threshold). Threshold will be applied later only for shared partners.
    string_all = pd.read_csv(STRING_LINKS, sep=" ")
    string_info = pd.read_csv(STRING_INFO, sep="\t")

    # Keep your original .str.replace behavior
    string_all["protein1"] = string_all["protein1"].astype(str).str.replace("9606.", "")
    string_all["protein2"] = string_all["protein2"].astype(str).str.replace("9606.", "")

    string_info["#string_protein_id"] = string_info["#string_protein_id"].astype(str).str.replace("9606.", "")
    string_info = string_info.iloc[:, :2].copy()

    # Add preferred names
    string_all = string_all.merge(string_info, left_on="protein1", right_on="#string_protein_id", how="left")
    string_all = string_all.rename(columns={"preferred_name": "gene_symbol_1"}).drop(columns=["#string_protein_id"])

    string_all = string_all.merge(string_info, left_on="protein2", right_on="#string_protein_id", how="left")
    string_all = string_all.rename(columns={"preferred_name": "gene_symbol_2"}).drop(columns=["#string_protein_id"])

    # Map using first-win HGNC lookup (original behavior)
    gene_lookup = build_hgnc_lookup_firstwin(hgnc_dict)

    string_all["hgnc_id_1"], string_all["ensembl_gene_id_1"], string_all["entrez_id_1"] = zip(
        *string_all["gene_symbol_1"].apply(lambda s: lookup_gene_info_firstwin(gene_lookup, str(s)))
    )
    string_all["hgnc_id_2"], string_all["ensembl_gene_id_2"], string_all["entrez_id_2"] = zip(
        *string_all["gene_symbol_2"].apply(lambda s: lookup_gene_info_firstwin(gene_lookup, str(s)))
    )

    return string_all


def build_string_unique_edges(string_df: pd.DataFrame) -> pd.DataFrame:
    # EXACT from your notebook (coercion + dropna before sorting)
    string_unique = (
        string_df[["entrez_id_1", "entrez_id_2"]]
        .rename(columns={"entrez_id_1": "A1_entrez", "entrez_id_2": "A2_entrez"})
        .drop_duplicates()
        .reset_index(drop=True)
    )

    string_unique["A1_entrez"] = pd.to_numeric(string_unique["A1_entrez"], errors="coerce").astype("Int64")
    string_unique["A2_entrez"] = pd.to_numeric(string_unique["A2_entrez"], errors="coerce").astype("Int64")

    print("N interactions:", string_unique.shape[0])

    string_unique = string_unique.dropna(subset=["A1_entrez", "A2_entrez"])

    string_unique = (
        pd.DataFrame(np.sort(string_unique[["A1_entrez", "A2_entrez"]], axis=1),
                     columns=["A1_entrez", "A2_entrez"])
        .drop_duplicates()
    )
    string_unique = string_unique[string_unique["A1_entrez"] != string_unique["A2_entrez"]]

    print("N interactions after sorting for unique pairs:", string_unique.shape[0])
    print("N genes in interaction map:", pd.concat([string_unique.A1_entrez, string_unique.A2_entrez]).nunique())

    return string_unique


def build_string_score_dict(string_all: pd.DataFrame) -> Dict[Tuple[object, object], float]:
    # Score dict built from FULL STRING (no threshold) -- matches your revised principle
    string_dict = {
        (row["ensembl_gene_id_1"], row["ensembl_gene_id_2"]): row["combined_score"]
        for _, row in string_all.iterrows()
    }
    string_dict.update({
        (row["ensembl_gene_id_2"], row["ensembl_gene_id_1"]): row["combined_score"]
        for _, row in string_all.iterrows()
    })
    return string_dict


# ==========================================================
# E) BIOGRID Physical: overwrite official symbol mapping + existence flags
# ==========================================================
def build_hgnc_lookup_biogrid_physical_overwrite(hgnc_dict) -> Dict[str, Tuple[object, object]]:
    gene_lookup = {}
    for entry in hgnc_dict:
        hgnc_id = entry.get("hgnc_id", np.nan)
        ensembl_id = entry.get("ensembl_gene_id", np.nan)

        symbol = normalize_symbol(entry.get("symbol"))
        if symbol:
            gene_lookup[symbol] = (hgnc_id, ensembl_id)

        if pd.notnull(entry.get("prev_symbol")):
            for prev in str(entry["prev_symbol"]).split("|"):
                key = normalize_symbol(prev)
                if key and key not in gene_lookup:
                    gene_lookup[key] = (hgnc_id, ensembl_id)

        if pd.notnull(entry.get("alias_symbol")):
            for alias in str(entry["alias_symbol"]).split("|"):
                key = normalize_symbol(alias)
                if key and key not in gene_lookup:
                    gene_lookup[key] = (hgnc_id, ensembl_id)

    return gene_lookup


def load_and_map_biogrid_physical(hgnc_dict) -> pd.DataFrame:
    biogrid = pd.read_csv(BIOGRID_ALL_TSV, sep="\t", low_memory=False)
    biogrid = biogrid[
        (biogrid["Experimental System Type"] == "physical") &
        (biogrid["Organism Name Interactor A"] == "Homo sapiens") &
        (biogrid["Organism Name Interactor B"] == "Homo sapiens")
    ].copy()

    biogrid["SortedInteractors"] = biogrid.apply(
        lambda row: "-".join(np.sort([row["Official Symbol Interactor A"], row["Official Symbol Interactor B"]])),
        axis=1
    )
    biogrid.drop_duplicates(subset="SortedInteractors", keep="first", inplace=True)
    biogrid.drop(columns="SortedInteractors", inplace=True)
    biogrid = biogrid.reset_index(drop=True)

    gene_lookup = build_hgnc_lookup_biogrid_physical_overwrite(hgnc_dict)

    def lookup(sym: str) -> Tuple[object, object]:
        key = normalize_symbol(sym)
        return gene_lookup.get(key, (np.nan, np.nan))

    biogrid["hgnc_id_1"], biogrid["ensembl_gene_id_1"] = zip(*biogrid["Official Symbol Interactor A"].apply(lookup))
    biogrid["hgnc_id_2"], biogrid["ensembl_gene_id_2"] = zip(*biogrid["Official Symbol Interactor B"].apply(lookup))

    return biogrid


def build_biogrid_exists_dict(biogrid: pd.DataFrame) -> Dict[Tuple[object, object], int]:
    biogrid_dict = {
        (row["ensembl_gene_id_1"], row["ensembl_gene_id_2"]): 1
        for _, row in biogrid.iterrows()
    }
    biogrid_dict.update({
        (row["ensembl_gene_id_2"], row["ensembl_gene_id_1"]): 1
        for _, row in biogrid.iterrows()
    })
    return biogrid_dict


# ==========================================================
# F) RUN (same pipeline, same files)
# ==========================================================
def main() -> None:
    ensure_dir(FEATURE_OUT_DIR)

    # Load main (Entrez and Ensembl columns used later)
    main_csv = pd.read_csv(MAIN_DATASET_CSV, low_memory=False)

    # Prepare gene pairs for shared interactors (same as your notebook)
    biomarker_query = main_csv[["entrez_id_biomarker", "entrez_id_query"]].rename(
        columns={"entrez_id_biomarker": "A1_entrez", "entrez_id_query": "A2_entrez"}
    )
    target1_query = main_csv[["entrez_id_target1", "entrez_id_query"]].rename(
        columns={"entrez_id_target1": "A1_entrez", "entrez_id_query": "A2_entrez"}
    )
    target2_query = main_csv[["entrez_id_target2", "entrez_id_query"]].rename(
        columns={"entrez_id_target2": "A1_entrez", "entrez_id_query": "A2_entrez"}
    )

    # --------------------------
    # BIOGRID-ALL shared interactors
    # --------------------------
    biogrid_raw = pd.read_csv(BIOGRID_ALL_TSV, sep="\t", low_memory=False)
    biogrid_unique = process_biogrid_ppi_data(biogrid_raw)

    compute_ppi_summary_for_pairs(biogrid_unique, biomarker_query).to_csv(OUT_FET_BIOGRID_BIOMARKER_QUERY, index=False)
    compute_ppi_summary_for_pairs(biogrid_unique, target1_query).to_csv(OUT_FET_BIOGRID_TARGET1_QUERY, index=False)
    compute_ppi_summary_for_pairs(biogrid_unique, target2_query).to_csv(OUT_FET_BIOGRID_TARGET2_QUERY, index=False)

    # --------------------------
    # HGNC dict once
    # --------------------------
    hgnc = pd.read_csv(HGNC_TSV, sep="\t", low_memory=False)
    columns_to_include = ["hgnc_id", "symbol", "prev_symbol", "ensembl_gene_id", "alias_symbol", "entrez_id"]
    hgnc_dict = hgnc[columns_to_include].to_dict(orient="records")

    # --------------------------
    # STRING: FULL table (NO threshold) for SCORE
    # --------------------------
    string_all = load_and_map_string_all(hgnc_dict)
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

    # --------------------------
    # BIOGRID Physical existence outputs
    # --------------------------
    main_csv3 = pd.read_csv(MAIN_DATASET_CSV, low_memory=False)
    biogrid_physical = load_and_map_biogrid_physical(hgnc_dict)
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

    main_csv3["BIOGRIDPhysicalInteractionQueryBiomarker"] = main_csv3.apply(check_bio_biomarker, axis=1)
    main_csv3["BIOGRIDPhysicalInteractionQueryTarget1"] = main_csv3.apply(check_bio_target1, axis=1)
    main_csv3["BIOGRIDPhysicalInteractionQueryTarget2"] = main_csv3.apply(check_bio_target2, axis=1)

    main_csv3["BIOGRIDPhysicalInteractionQueryTarget"] = (
        main_csv3["BIOGRIDPhysicalInteractionQueryTarget1"] | main_csv3["BIOGRIDPhysicalInteractionQueryTarget2"]
    ).astype(int)

    main_csv3[["ensembl_gene_id_query", "ensembl_gene_id_biomarker", "BIOGRIDPhysicalInteractionQueryBiomarker"]].to_csv(
        OUT_BIOGRID_BIOMARKER_QUERY, index=False
    )
    main_csv3[["ensembl_gene_id_query", "ensembl_gene_id_target1", "BIOGRIDPhysicalInteractionQueryTarget"]].to_csv(
        OUT_BIOGRID_TARGET_QUERY, index=False
    )
        # --------------------------
    # Biomarker TSG / oncogene label (Cancer Gene Census)  [ADDED]
    # --------------------------
    try:
        cancer_gene_census = pd.read_csv(CANCER_GENE_CENSUS_CSV, low_memory=False)
        biomarker_type = classify_biomarker_tsg_oncogene(main_csv3, cancer_gene_census)

        # Output matches notebook intent: only Biomarker + label
        biomarker_type.to_csv(OUT_BIOMARKER_TSG_LABEL, index=False)
        print("[OK] Biomarker TSG/oncogene labels written:", OUT_BIOMARKER_TSG_LABEL)

    except FileNotFoundError:
        print(f"[WARN] Cancer Gene Census file not found: {CANCER_GENE_CENSUS_CSV} (skipping TSG/oncogene labels)")
    except Exception as e:
        print(f"[WARN] Failed to compute Biomarker TSG/oncogene labels: {e}")

    print("[OK] Finished. Outputs written to:", FEATURE_OUT_DIR)


main()