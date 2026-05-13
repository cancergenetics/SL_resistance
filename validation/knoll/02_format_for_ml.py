"""Step 02 — Reformat Knoll labels into the main ML dataset schema.

Input:  data/labels_AvgDiff.csv  (query_gene = human symbol, label, ...)
Output: data/features_main_schema.csv

Schema: 30-column PredictingSLResistanceFeatures_main.csv format.
Feature columns left NaN; feature-extraction pipeline (03-06) fills them.
Target2 = NaN (PRMT5 is single drug target).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BIOMARKER_GENE,
    DATA,
    FEATURES_MAIN_SCHEMA_CSV,
    HGNC_TSV,
    LABEL_ONLY_FEATURE_COLS,
    LABELS_HUMAN_CSV,
    MAIN_SCHEMA_CSV,
    SCREEN_AVG,
    SL_PAIR,
    TARGET1_GENE,
    TARGET2_GENE,
)


def build_hgnc_lookup(genes: set[str]) -> pd.DataFrame:
    hgnc = pd.read_csv(HGNC_TSV, sep="\t", low_memory=False)
    approved = hgnc.set_index("symbol")[["hgnc_id", "entrez_id", "ensembl_gene_id"]]

    alias_map: dict[str, str] = {}
    prev_map:  dict[str, str] = {}
    for col, store in [("alias_symbol", alias_map), ("prev_symbol", prev_map)]:
        for sym, val in zip(hgnc["symbol"], hgnc[col]):
            if pd.isna(val):
                continue
            for alt in str(val).split("|"):
                alt = alt.strip()
                if alt and alt not in store:
                    store[alt] = sym

    rows = []
    for g in sorted(genes):
        if g in approved.index:
            r = approved.loc[g]
        elif g in alias_map and alias_map[g] in approved.index:
            r = approved.loc[alias_map[g]]
        elif g in prev_map and prev_map[g] in approved.index:
            r = approved.loc[prev_map[g]]
        else:
            r = pd.Series({"hgnc_id": np.nan, "entrez_id": np.nan, "ensembl_gene_id": np.nan})
        rows.append((g, r["hgnc_id"], r["entrez_id"], r["ensembl_gene_id"]))

    return pd.DataFrame(
        rows, columns=["gene", "hgnc_id", "entrez_id", "ensembl_gene_id"]
    ).set_index("gene")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    column_order = list(pd.read_csv(MAIN_SCHEMA_CSV, nrows=1).columns)
    labels = pd.read_csv(LABELS_HUMAN_CSV)

    all_genes: set[str] = {BIOMARKER_GENE, TARGET1_GENE}
    if TARGET2_GENE:
        all_genes.add(TARGET2_GENE)
    all_genes |= set(labels["query_gene"].dropna().astype(str))

    print(f"Building HGNC lookup for {len(all_genes)} unique genes …")
    lookup = build_hgnc_lookup(all_genes)

    out = pd.DataFrame(index=labels.index)
    out["Screen"]    = SCREEN_AVG
    out["SL_Pair"]   = SL_PAIR
    out["Query"]     = labels["query_gene"].values
    out["Biomarker"] = BIOMARKER_GENE
    out["Target1"]   = TARGET1_GENE
    out["Target2"]   = TARGET2_GENE if TARGET2_GENE else np.nan

    def pull(series: pd.Series, field: str) -> np.ndarray:
        return series.map(lookup[field]).values

    out["entrez_id_query"]           = pull(labels["query_gene"], "entrez_id")
    out["hgnc_id_query"]             = pull(labels["query_gene"], "hgnc_id")
    out["ensembl_gene_id_query"]     = pull(labels["query_gene"], "ensembl_gene_id")
    out["entrez_id_biomarker"]       = lookup.at[BIOMARKER_GENE, "entrez_id"]
    out["hgnc_id_biomarker"]         = lookup.at[BIOMARKER_GENE, "hgnc_id"]
    out["ensembl_gene_id_biomarker"] = lookup.at[BIOMARKER_GENE, "ensembl_gene_id"]
    out["entrez_id_target1"]         = lookup.at[TARGET1_GENE, "entrez_id"]
    out["hgnc_id_target1"]           = lookup.at[TARGET1_GENE, "hgnc_id"]
    out["ensembl_gene_id_target1"]   = lookup.at[TARGET1_GENE, "ensembl_gene_id"]

    if TARGET2_GENE and TARGET2_GENE in lookup.index:
        out["entrez_id_target2"]         = lookup.at[TARGET2_GENE, "entrez_id"]
        out["hgnc_id_target2"]           = lookup.at[TARGET2_GENE, "hgnc_id"]
        out["ensembl_gene_id_target2"]   = lookup.at[TARGET2_GENE, "ensembl_gene_id"]
    else:
        out["entrez_id_target2"]         = np.nan
        out["hgnc_id_target2"]           = np.nan
        out["ensembl_gene_id_target2"]   = np.nan

    out["Class"] = labels["label"].values
    for c in LABEL_ONLY_FEATURE_COLS:
        out[c] = np.nan

    out = out[column_order]
    out.to_csv(FEATURES_MAIN_SCHEMA_CSV, index=False)

    counts  = out["Class"].value_counts()
    missing = [g for g in labels["query_gene"].unique()
               if g in lookup.index and pd.isna(lookup.at[g, "hgnc_id"])]
    print(f"rows:           {len(out)}")
    print(f"Resistance:     {counts.get('Resistance', 0)}")
    print(f"Non-Resistance: {counts.get('Non-Resistance', 0)}")
    print(f"MTAP  HGNC: {lookup.at[BIOMARKER_GENE, 'hgnc_id']}  "
          f"entrez: {lookup.at[BIOMARKER_GENE, 'entrez_id']}")
    print(f"PRMT5 HGNC: {lookup.at[TARGET1_GENE, 'hgnc_id']}  "
          f"entrez: {lookup.at[TARGET1_GENE, 'entrez_id']}")
    print(f"HGNC-unresolved query genes: {len(missing)}")
    print(f"wrote {FEATURES_MAIN_SCHEMA_CSV}")


if __name__ == "__main__":
    main()