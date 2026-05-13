#!/usr/bin/env python
# coding: utf-8


import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

# =========================
# Configuration
# =========================
CONFIG = {
    "dataset_dir": "datasets",
    "feature_dir": "feature_output",
    "pairs_xlsx": "biomarker_target_genes.xlsx",
    "hgnc_file": "../input_data/HGNC/hgnc_complete_set.txt",
    "biogrid_all_tsv": "../input_data/BIOGRID/BIOGRID-ALL-4.4.241.tab3.txt",
    "biogrid_mv_tsv": "../input_data/BIOGRID/BIOGRID-MV-Physical-4.4.229.tab3.txt",
    "string_links": "../input_data/STRING/9606.protein.links.detailed.v12.0.txt",
    "string_info": "../input_data/STRING/9606.protein.info.v12.0.txt",
    "cancer_gene_census_csv": "../input_data/Cancer_Gene_Census/Cancer_gene_census_data.csv",
    "string_score_threshold": 400,
}



# =========================
# Shared utilities
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_symbol(s):
    return s.strip().upper() if isinstance(s, str) else None


def load_combined_clinical_main(dataset_dir: str, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Load each per-pair main CSV produced by notebook 1 and concatenate into one frame."""
    frames: List[pd.DataFrame] = []
    for _, row in pairs_df.iterrows():
        sl_pair = f"{row['Biomarker']}_{row['Target']}"
        path = os.path.join(dataset_dir, f"PredictingSLResistanceFeatures_{sl_pair}_main.csv")
        frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def load_hgnc_records(hgnc_file: str) -> List[dict]:
    hgnc = pd.read_csv(hgnc_file, sep="\t", low_memory=False)
    cols = ["hgnc_id", "symbol", "prev_symbol", "ensembl_gene_id", "alias_symbol", "entrez_id"]
    return hgnc[cols].to_dict(orient="records")


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


def compute_ppi_summary_for_pairs(ppi: pd.DataFrame, gene_pairs: pd.DataFrame) -> pd.DataFrame:
    """Shared-interactors Fisher Exact Test — reused for BIOGRID-ALL and STRING."""
    ppi_symmetric = pd.concat([
        ppi,
        ppi.rename(columns={"A1_entrez": "A2_entrez", "A2_entrez": "A1_entrez"}),
    ]).reset_index(drop=True)
    assert ppi_symmetric.shape[0] == ppi.shape[0] * 2

    gene_ppi = pd.merge(
        gene_pairs[["A1_entrez", "A2_entrez"]],
        ppi_symmetric,
        how="left",
        indicator="interact",
    )
    gene_ppi.interact = gene_ppi.interact == "both"
    assert gene_ppi.shape[0] == gene_pairs.shape[0]
    print("N. gene pairs that interact:", int(gene_ppi.interact.sum()))

    ppi_per_gene = (
        ppi_symmetric.groupby("A1_entrez")
        .agg({"A2_entrez": set})
        .reset_index()
        .rename(columns={"A1_entrez": "gene", "A2_entrez": "ppi"})
    )
    df = pd.merge(gene_ppi, ppi_per_gene.rename(columns={"gene": "A1_entrez", "ppi": "A1_ppi"}), how="left")
    df = pd.merge(df, ppi_per_gene.rename(columns={"gene": "A2_entrez", "ppi": "A2_ppi"}), how="left")

    df["A1_ppi"] = df["A1_ppi"].apply(lambda d: d if isinstance(d, set) else set())
    df["A2_ppi"] = df["A2_ppi"].apply(lambda d: d if isinstance(d, set) else set())
    df["A1_ppi"] = df.apply(lambda x: x.A1_ppi - {x.A2_entrez}, axis=1)
    df["A2_ppi"] = df.apply(lambda x: x.A2_ppi - {x.A1_entrez}, axis=1)
    df["n_A1_ppi"] = df.A1_ppi.apply(len)
    df["n_A2_ppi"] = df.A2_ppi.apply(len)
    df["shared_ppi"] = df.apply(lambda x: x.A1_ppi.intersection(x.A2_ppi), axis=1)
    df["n_total_ppi"] = df.apply(lambda x: len(x.A1_ppi.union(x.A2_ppi)), axis=1)
    df["n_shared_ppi"] = df.shared_ppi.apply(len)

    def calc_jaccard_index(x):
        if x.n_shared_ppi == 0:
            return 0
        return x.n_shared_ppi / ((x.n_A1_ppi + x.n_A2_ppi) - x.n_shared_ppi)

    df["shared_ppi_jaccard_idx"] = df.apply(calc_jaccard_index, axis=1)

    N = len(pd.concat([ppi.A1_entrez, ppi.A2_entrez]).unique())

    def calc_fet_shared_ppi(x):
        ctab = pd.DataFrame(
            {
                "A2": [x.n_shared_ppi, x.n_A2_ppi - x.n_shared_ppi],
                "NA2": [x.n_A1_ppi - x.n_shared_ppi, N - x.n_total_ppi],
            },
            index=["A1", "NA1"],
        )
        OR, pval = stats.fisher_exact(ctab)
        if pval == 0:
            pval = np.nextafter(0, 1)
        log_pval = (-np.log10(pval)) if pval != 1 else 0
        log_pval = -log_pval if OR < 1 else log_pval
        return log_pval

    df["fet_ppi_overlap"] = df.apply(calc_fet_shared_ppi, axis=1)
    df = df.drop(columns=["A1_ppi", "A2_ppi", "n_A1_ppi", "n_A2_ppi"])
    return df



# =========================
# HGNC lookup builders (copy-adapted from 7_feature_extraction_ppi_based_refractored.ipynb)
# =========================
def build_hgnc_lookup_firstwin(hgnc_dict) -> Dict[str, Tuple[object, object, str]]:
    gene_lookup: Dict[str, Tuple[object, object, str]] = {}
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


def lookup_gene_info_firstwin(gene_lookup, gene_symbol):
    key = normalize_symbol(gene_symbol)
    return gene_lookup.get(key, (np.nan, np.nan, "NA"))


def build_hgnc_lookup_biogrid_mv(hgnc_dict) -> Dict[str, Tuple[object, object]]:
    gene_lookup: Dict[str, Tuple[object, object]] = {}
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



# =========================
# BIOGRID-ALL shared interactors + FET
# =========================
def process_biogrid_ppi_data(data: pd.DataFrame) -> pd.DataFrame:
    hsapien = 9606
    biogrid_ppi = data[
        (data["Organism ID Interactor A"] == hsapien)
        & (data["Organism ID Interactor B"] == hsapien)
        & (data["Experimental System Type"] == "physical")
    ]
    biogrid_ppi = biogrid_ppi[[
        "Entrez Gene Interactor A", "Entrez Gene Interactor B", "Experimental System",
    ]].rename(columns={
        "Entrez Gene Interactor A": "A1_entrez",
        "Entrez Gene Interactor B": "A2_entrez",
        "Experimental System": "experimental_system",
    })
    biogrid_ppi = (
        biogrid_ppi.drop_duplicates()
        .reset_index(drop=True)
        .astype({"A1_entrez": "int", "A2_entrez": "int"})
    )
    biogrid_ppi = biogrid_ppi[biogrid_ppi["A1_entrez"] != biogrid_ppi["A2_entrez"]].reset_index(drop=True)

    biogrid_unique = pd.DataFrame(
        np.sort(biogrid_ppi[["A1_entrez", "A2_entrez"]], axis=1),
        columns=["A1_entrez", "A2_entrez"],
    ).drop_duplicates()
    return biogrid_unique.reset_index(drop=True)


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



# =========================
# STRING loaders + score dict + shared-partners FET
# =========================
def load_and_map_string_all(string_links_path: str, string_info_path: str, hgnc_dict) -> pd.DataFrame:
    string_all = pd.read_csv(string_links_path, sep=" ")
    string_info = pd.read_csv(string_info_path, sep="\t")

    string_all["protein1"] = string_all["protein1"].astype(str).str.replace("9606.", "", regex=False)
    string_all["protein2"] = string_all["protein2"].astype(str).str.replace("9606.", "", regex=False)

    string_info["#string_protein_id"] = string_info["#string_protein_id"].astype(str).str.replace("9606.", "", regex=False)
    string_info = string_info.iloc[:, :2].copy()

    string_all = string_all.merge(string_info, left_on="protein1", right_on="#string_protein_id", how="left")
    string_all = string_all.rename(columns={"preferred_name": "gene_symbol_1"}).drop(columns=["#string_protein_id"])

    string_all = string_all.merge(string_info, left_on="protein2", right_on="#string_protein_id", how="left")
    string_all = string_all.rename(columns={"preferred_name": "gene_symbol_2"}).drop(columns=["#string_protein_id"])

    gene_lookup = build_hgnc_lookup_firstwin(hgnc_dict)

    string_all["hgnc_id_1"], string_all["ensembl_gene_id_1"], string_all["entrez_id_1"] = zip(
        *string_all["gene_symbol_1"].apply(lambda s: lookup_gene_info_firstwin(gene_lookup, str(s)))
    )
    string_all["hgnc_id_2"], string_all["ensembl_gene_id_2"], string_all["entrez_id_2"] = zip(
        *string_all["gene_symbol_2"].apply(lambda s: lookup_gene_info_firstwin(gene_lookup, str(s)))
    )
    return string_all


def build_string_score_dict(string_all: pd.DataFrame) -> Dict[Tuple[object, object], float]:
    d = {
        (row["ensembl_gene_id_1"], row["ensembl_gene_id_2"]): row["combined_score"]
        for _, row in string_all.iterrows()
    }
    d.update({
        (row["ensembl_gene_id_2"], row["ensembl_gene_id_1"]): row["combined_score"]
        for _, row in string_all.iterrows()
    })
    return d


def build_string_unique_edges(string_df: pd.DataFrame) -> pd.DataFrame:
    string_unique = (
        string_df[["entrez_id_1", "entrez_id_2"]]
        .rename(columns={"entrez_id_1": "A1_entrez", "entrez_id_2": "A2_entrez"})
        .drop_duplicates()
        .reset_index(drop=True)
    )
    string_unique["A1_entrez"] = pd.to_numeric(string_unique["A1_entrez"], errors="coerce").astype("Int64")
    string_unique["A2_entrez"] = pd.to_numeric(string_unique["A2_entrez"], errors="coerce").astype("Int64")
    string_unique = string_unique.dropna(subset=["A1_entrez", "A2_entrez"])
    string_unique = (
        pd.DataFrame(
            np.sort(string_unique[["A1_entrez", "A2_entrez"]], axis=1),
            columns=["A1_entrez", "A2_entrez"],
        )
        .drop_duplicates()
    )
    string_unique = string_unique[string_unique["A1_entrez"] != string_unique["A2_entrez"]].reset_index(drop=True)
    return string_unique


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
    5_feature_merge.ipynb to match main notebook 10."""
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



# =========================
# BIOGRID-MV physical binary existence
# =========================
def load_and_map_biogrid_mv(biogrid_mv_path: str, hgnc_dict) -> pd.DataFrame:
    biogrid = pd.read_csv(biogrid_mv_path, sep="\t")
    biogrid = biogrid[
        (biogrid["Organism Name Interactor A"] == "Homo sapiens")
        & (biogrid["Organism Name Interactor B"] == "Homo sapiens")
    ].copy()

    biogrid["SortedInteractors"] = biogrid.apply(
        lambda row: "-".join(np.sort([row["Official Symbol Interactor A"], row["Official Symbol Interactor B"]])),
        axis=1,
    )
    biogrid.drop_duplicates(subset="SortedInteractors", keep="first", inplace=True)
    biogrid.drop(columns="SortedInteractors", inplace=True)
    biogrid = biogrid.reset_index(drop=True)

    gene_lookup = build_hgnc_lookup_biogrid_mv(hgnc_dict)

    def lookup(sym):
        key = normalize_symbol(sym)
        return gene_lookup.get(key, (np.nan, np.nan))

    biogrid["hgnc_id_1"], biogrid["ensembl_gene_id_1"] = zip(*biogrid["Official Symbol Interactor A"].apply(lookup))
    biogrid["hgnc_id_2"], biogrid["ensembl_gene_id_2"] = zip(*biogrid["Official Symbol Interactor B"].apply(lookup))
    return biogrid


def build_biogrid_exists_dict(biogrid: pd.DataFrame) -> Dict[Tuple[object, object], int]:
    d = {(row["ensembl_gene_id_1"], row["ensembl_gene_id_2"]): 1 for _, row in biogrid.iterrows()}
    d.update({(row["ensembl_gene_id_2"], row["ensembl_gene_id_1"]): 1 for _, row in biogrid.iterrows()})
    return d


def run_biogrid_mv_existence(
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
# Cancer Gene Census TSG / oncogene classification
# =========================
def classify_biomarker_tsg_oncogene(combined: pd.DataFrame, census_df: pd.DataFrame) -> pd.DataFrame:
    tmp = combined[["SL_Pair", "Biomarker"]].copy()
    tmp["Biomarker"] = tmp["Biomarker"].astype(str)

    census = census_df.copy()
    census["Gene Symbol"] = census["Gene Symbol"].astype(str)
    role_mapping = census.set_index("Gene Symbol")["Role in Cancer"].to_dict()

    def classify_gene(gene_symbol):
        role = role_mapping.get(gene_symbol, None)
        if role:
            if "TSG" in role:
                return 1
            elif "oncogene" in role:
                return 0
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

hgnc_records = load_hgnc_records(CONFIG["hgnc_file"])
print(f"[info] HGNC records: {len(hgnc_records):,}")

# --- BIOGRID-ALL shared interactors FET ---
biogrid_all_raw = pd.read_csv(CONFIG["biogrid_all_tsv"], sep="\t", low_memory=False)
biogrid_unique = process_biogrid_ppi_data(biogrid_all_raw)
print(f"[info] BIOGRID-ALL unique physical edges: {len(biogrid_unique):,}")

run_biogrid_fet_per_partner(biogrid_unique, combined, "biomarker", CONFIG["feature_dir"])
run_biogrid_fet_per_partner(biogrid_unique, combined, "target1", CONFIG["feature_dir"])

# --- STRING shared interactors FET + raw scores ---
string_all = load_and_map_string_all(CONFIG["string_links"], CONFIG["string_info"], hgnc_records)
print(f"[info] STRING edges (all): {len(string_all):,}")

string_score_dict = build_string_score_dict(string_all)

string_filtered = string_all[string_all["combined_score"] >= CONFIG["string_score_threshold"]]
string_unique_edges = build_string_unique_edges(string_filtered)
print(f"[info] STRING unique edges >= {CONFIG['string_score_threshold']}: {len(string_unique_edges):,}")

run_string_fet(string_unique_edges, combined, "biomarker", CONFIG["feature_dir"])
run_string_fet(string_unique_edges, combined, "target1", CONFIG["feature_dir"])
run_string_scores(combined, string_score_dict, CONFIG["feature_dir"])

# --- BIOGRID-MV physical binary existence ---
biogrid_mv = load_and_map_biogrid_mv(CONFIG["biogrid_mv_tsv"], hgnc_records)
biogrid_exists_dict = build_biogrid_exists_dict(biogrid_mv)
run_biogrid_mv_existence(combined, biogrid_exists_dict, CONFIG["feature_dir"])

# --- Cancer Gene Census TSG / oncogene labels ---
census_df = pd.read_csv(CONFIG["cancer_gene_census_csv"], low_memory=False)
run_biomarker_type(combined, census_df, CONFIG["feature_dir"])

print("[OK] 2_feature_extraction_ppi_based complete")