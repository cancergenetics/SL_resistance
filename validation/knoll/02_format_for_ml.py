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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for lib
from lib.hgnc_lookup import build_hgnc_lookup
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


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    column_order = list(pd.read_csv(MAIN_SCHEMA_CSV, nrows=1).columns)
    labels = pd.read_csv(LABELS_HUMAN_CSV)

    all_genes: set[str] = {BIOMARKER_GENE, TARGET1_GENE}
    if TARGET2_GENE:
        all_genes.add(TARGET2_GENE)
    all_genes |= set(labels["query_gene"].dropna().astype(str))

    print(f"Building HGNC lookup for {len(all_genes)} unique genes …")
    lookup = build_hgnc_lookup(all_genes, HGNC_TSV)   # shared 3-tier resolver

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

    # No third target for this pair (MTAP-PRMT5). Target3 columns were added to the
    # main schema by the AKT 3-target change (PTEN_AKT -> AKT1/AKT2/AKT3); fill NaN.
    out["Target3"]                   = np.nan
    out["entrez_id_target3"]         = np.nan
    out["hgnc_id_target3"]           = np.nan
    out["ensembl_gene_id_target3"]   = np.nan

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