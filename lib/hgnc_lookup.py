"""Shared library — HGNC symbol resolution.

Resolves human gene symbols to HGNC ID + Entrez + Ensembl gene ID. Tries:
  1. Approved `symbol` exact match
  2. `alias_symbol` lookup
  3. `prev_symbol` lookup

Public entry point: build_hgnc_lookup(genes, hgnc_tsv) -> DataFrame indexed by gene
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_hgnc_lookup(genes, hgnc_tsv: Path) -> pd.DataFrame:
    hgnc = pd.read_csv(hgnc_tsv, sep="\t", low_memory=False)
    approved = hgnc.set_index("symbol")[["hgnc_id", "entrez_id", "ensembl_gene_id"]]

    alias_map = {}
    prev_map = {}
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
