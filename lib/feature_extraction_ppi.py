"""Shared library — PPI feature extraction CORE (single source of truth).

This module holds the gene-ID mapping + computation building blocks shared by the
main pipeline (`07_feature_extraction_ppi_based.py`) and the clinical-trials
pipeline (`clinical_trials/02_feature_extraction_ppi_based.py`). Each of those
scripts keeps only its own I/O orchestration (the main pipeline writes one
combined feature table; clinical writes per-SL_pair files) and imports the
functions below for the actual computation.

Partner symbols (STRING `preferred_name`, BioGRID `Official Symbol`) are resolved
to HGNC/Ensembl/Entrez IDs through the shared strict 3-tier resolver
`lib.hgnc_lookup.build_hgnc_lookup` (symbol > prev_symbol > alias, case-insensitive)
— NOT the old first-win lookup, which mis-mapped alias collisions (e.g. STRING
"ATR" → ANTXR1).

Provides:
  - classify_biomarker_tsg_oncogene
  - process_biogrid_ppi_data          (BIOGRID-ALL entrez shared-interactor edges)
  - compute_ppi_summary_for_pairs     (shared-interactor FET feature)
  - load_and_map_string_all / build_string_unique_edges / build_string_score_dict
  - load_and_map_biogrid_physical / build_biogrid_exists_dict
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from lib.hgnc_lookup import build_hgnc_lookup

STRING_SCORE_THRESHOLD = 400


def normalize_symbol(s: Optional[str]) -> Optional[str]:
    return s.strip().upper() if isinstance(s, str) else None


# ----------------------------------------------------------------------------
# Shared 3-tier symbol -> IDs mapper (replaces the old first-win lookups)
# ----------------------------------------------------------------------------
def make_id_mapper(symbols, hgnc_tsv):
    """Return a function symbol(str) -> (hgnc_id, ensembl_gene_id, entrez_id) built
    over `symbols` via the strict 3-tier resolver lib.hgnc_lookup.build_hgnc_lookup.
    Unmapped/blank symbols return (nan, nan, "NA")."""
    ser = pd.Series(list(symbols)).dropna().astype(str)
    uniq = [s for s in pd.unique(ser) if s and s.lower() != "nan"]
    lkp = build_hgnc_lookup(uniq, hgnc_tsv)   # DataFrame indexed by gene

    def ids3(symbol):
        if isinstance(symbol, str) and symbol in lkp.index:
            r = lkp.loc[symbol]
            ent = r["entrez_id"]
            return (r["hgnc_id"], r["ensembl_gene_id"], ("NA" if pd.isna(ent) else ent))
        return (np.nan, np.nan, "NA")

    return ids3


# ----------------------------------------------------------------------------
# Cancer Gene Census — biomarker TSG/oncogene label
# ----------------------------------------------------------------------------
def classify_biomarker_tsg_oncogene(main_df: pd.DataFrame, census_df: pd.DataFrame) -> pd.DataFrame:
    if "Biomarker" not in main_df.columns:
        raise KeyError("'Biomarker' column not found in MAIN dataset")
    if "Gene Symbol" not in census_df.columns or "Role in Cancer" not in census_df.columns:
        raise KeyError("Cancer Gene Census file must contain 'Gene Symbol' and 'Role in Cancer'")

    tmp = main_df[["Biomarker"]].copy()
    tmp["Biomarker"] = tmp["Biomarker"].astype(str)
    role_mapping = census_df.assign(**{"Gene Symbol": census_df["Gene Symbol"].astype(str)}) \
        .set_index("Gene Symbol")["Role in Cancer"].to_dict()

    def classify_gene(sym: str):
        role = role_mapping.get(sym)
        if role and not pd.isna(role):
            if "TSG" in role:
                return 1
            if "oncogene" in role:
                return 0
        return np.nan

    tmp["TSG_Label"] = tmp["Biomarker"].apply(classify_gene)
    return tmp


# ----------------------------------------------------------------------------
# BIOGRID-ALL (Entrez) shared interactors -> FET feature
# ----------------------------------------------------------------------------
def process_biogrid_ppi_data(data: pd.DataFrame) -> pd.DataFrame:
    hsapien = 9606
    biogrid_ppi = data[
        (data["Organism ID Interactor A"] == hsapien)
        & (data["Organism ID Interactor B"] == hsapien)
        & (data["Experimental System Type"] == "physical")
    ][["Entrez Gene Interactor A", "Entrez Gene Interactor B", "Experimental System"]].rename(
        columns={
            "Entrez Gene Interactor A": "A1_entrez",
            "Entrez Gene Interactor B": "A2_entrez",
            "Experimental System": "experimental_system",
        }
    ).drop_duplicates().reset_index(drop=True).astype({"A1_entrez": "int", "A2_entrez": "int"})
    biogrid_ppi = biogrid_ppi[biogrid_ppi["A1_entrez"] != biogrid_ppi["A2_entrez"]].reset_index(drop=True)

    biogrid_unique = pd.DataFrame(
        np.sort(biogrid_ppi[["A1_entrez", "A2_entrez"]], axis=1),
        columns=["A1_entrez", "A2_entrez"],
    ).drop_duplicates().reset_index(drop=True)
    return biogrid_unique


def compute_ppi_summary_for_pairs(ppi: pd.DataFrame, gene_pairs: pd.DataFrame) -> pd.DataFrame:
    ppi_symmetric = pd.concat([ppi, ppi.rename(columns={"A1_entrez": "A2_entrez", "A2_entrez": "A1_entrez"})]).reset_index(drop=True)
    assert ppi_symmetric.shape[0] == ppi.shape[0] * 2

    gene_ppi = pd.merge(gene_pairs[["A1_entrez", "A2_entrez"]], ppi_symmetric, how="left", indicator="interact")
    gene_ppi.interact = gene_ppi.interact == "both"
    print("N. gene pairs that interact:", sum(gene_ppi.interact))

    ppi_per_gene = (
        ppi_symmetric.groupby("A1_entrez").agg({"A2_entrez": set}).reset_index()
        .rename(columns={"A1_entrez": "gene", "A2_entrez": "ppi"})
    )

    df = pd.merge(gene_ppi, ppi_per_gene.rename(columns={"gene": "A1_entrez", "ppi": "A1_ppi"}), how="left")
    df = pd.merge(df, ppi_per_gene.rename(columns={"gene": "A2_entrez", "ppi": "A2_ppi"}), how="left")

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

    def calc_fet_shared_ppi(x):
        ctab = pd.DataFrame(
            {"A2": [x.n_shared_ppi, x.n_A2_ppi - x.n_shared_ppi],
             "NA2": [x.n_A1_ppi - x.n_shared_ppi, N - x.n_total_ppi]},
            index=["A1", "NA1"],
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


# ----------------------------------------------------------------------------
# STRING: full table (no threshold) for SCORE; >=400 for shared partners
# ----------------------------------------------------------------------------
def load_and_map_string_all(string_links, string_info_path, hgnc_tsv) -> pd.DataFrame:
    string_all = pd.read_csv(string_links, sep=" ")
    string_info = pd.read_csv(string_info_path, sep="\t")
    string_all["protein1"] = string_all["protein1"].astype(str).str.replace("9606.", "")
    string_all["protein2"] = string_all["protein2"].astype(str).str.replace("9606.", "")
    string_info["#string_protein_id"] = string_info["#string_protein_id"].astype(str).str.replace("9606.", "")
    string_info = string_info.iloc[:, :2].copy()

    string_all = string_all.merge(string_info, left_on="protein1", right_on="#string_protein_id", how="left")
    string_all = string_all.rename(columns={"preferred_name": "gene_symbol_1"}).drop(columns=["#string_protein_id"])
    string_all = string_all.merge(string_info, left_on="protein2", right_on="#string_protein_id", how="left")
    string_all = string_all.rename(columns={"preferred_name": "gene_symbol_2"}).drop(columns=["#string_protein_id"])

    ids3 = make_id_mapper(
        pd.concat([string_all["gene_symbol_1"], string_all["gene_symbol_2"]], ignore_index=True),
        hgnc_tsv,
    )
    string_all["hgnc_id_1"], string_all["ensembl_gene_id_1"], string_all["entrez_id_1"] = zip(
        *string_all["gene_symbol_1"].astype(str).map(ids3)
    )
    string_all["hgnc_id_2"], string_all["ensembl_gene_id_2"], string_all["entrez_id_2"] = zip(
        *string_all["gene_symbol_2"].astype(str).map(ids3)
    )
    return string_all


def build_string_unique_edges(string_df: pd.DataFrame) -> pd.DataFrame:
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
    string_unique = pd.DataFrame(
        np.sort(string_unique[["A1_entrez", "A2_entrez"]], axis=1), columns=["A1_entrez", "A2_entrez"]
    ).drop_duplicates()
    string_unique = string_unique[string_unique["A1_entrez"] != string_unique["A2_entrez"]]
    print("N interactions after sorting for unique pairs:", string_unique.shape[0])
    print("N genes in interaction map:", pd.concat([string_unique.A1_entrez, string_unique.A2_entrez]).nunique())
    return string_unique


def build_string_score_dict(string_all: pd.DataFrame) -> Dict[Tuple[object, object], float]:
    string_dict = {
        (row["ensembl_gene_id_1"], row["ensembl_gene_id_2"]): row["combined_score"]
        for _, row in string_all.iterrows()
    }
    string_dict.update({
        (row["ensembl_gene_id_2"], row["ensembl_gene_id_1"]): row["combined_score"]
        for _, row in string_all.iterrows()
    })
    return string_dict


# ----------------------------------------------------------------------------
# BIOGRID Physical: official-symbol mapping + existence flags
# ----------------------------------------------------------------------------
def load_and_map_biogrid_physical(biogrid_all_tsv, hgnc_tsv) -> pd.DataFrame:
    biogrid = pd.read_csv(biogrid_all_tsv, sep="\t", low_memory=False)
    biogrid = biogrid[
        (biogrid["Experimental System Type"] == "physical")
        & (biogrid["Organism Name Interactor A"] == "Homo sapiens")
        & (biogrid["Organism Name Interactor B"] == "Homo sapiens")
    ].copy()
    biogrid["SortedInteractors"] = biogrid.apply(
        lambda row: "-".join(np.sort([row["Official Symbol Interactor A"], row["Official Symbol Interactor B"]])),
        axis=1,
    )
    biogrid.drop_duplicates(subset="SortedInteractors", keep="first", inplace=True)
    biogrid.drop(columns="SortedInteractors", inplace=True)
    biogrid = biogrid.reset_index(drop=True)

    ids3 = make_id_mapper(
        pd.concat([biogrid["Official Symbol Interactor A"], biogrid["Official Symbol Interactor B"]], ignore_index=True),
        hgnc_tsv,
    )

    def ids2(sym):
        h, e, _ = ids3(sym)
        return (h, e)

    biogrid["hgnc_id_1"], biogrid["ensembl_gene_id_1"] = zip(*biogrid["Official Symbol Interactor A"].astype(str).map(ids2))
    biogrid["hgnc_id_2"], biogrid["ensembl_gene_id_2"] = zip(*biogrid["Official Symbol Interactor B"].astype(str).map(ids2))
    return biogrid


def build_biogrid_exists_dict(biogrid: pd.DataFrame) -> Dict[Tuple[object, object], int]:
    d = {(r["ensembl_gene_id_1"], r["ensembl_gene_id_2"]): 1 for _, r in biogrid.iterrows()}
    d.update({(r["ensembl_gene_id_2"], r["ensembl_gene_id_1"]): 1 for _, r in biogrid.iterrows()})
    return d
