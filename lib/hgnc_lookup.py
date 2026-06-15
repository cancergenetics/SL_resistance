"""Shared library — HGNC symbol resolution (single source of truth).

Resolves human gene symbols to HGNC ID + Entrez + Ensembl gene ID with strict
3-tier precedence:
  1. Approved `symbol`
  2. `prev_symbol` (former approved name)
  3. `alias_symbol` (informal alias)

A former official (previous) symbol takes precedence over an informal alias, so a
symbol that is a previous symbol of one gene and an alias of another resolves to
the gene for which it was the official name. A canonical symbol is never shadowed
by another gene's previous/alias name. Matching is case-insensitive (symbols are
normalized strip+upper on both sides), so callers may pass any casing.

Public entry point: build_hgnc_lookup(genes, hgnc_tsv) -> DataFrame indexed by the
input gene (as passed), columns [hgnc_id, entrez_id, ensembl_gene_id].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _norm(s):
    return s.strip().upper() if isinstance(s, str) else None


def build_hgnc_lookup(genes, hgnc_tsv: Path) -> pd.DataFrame:
    hgnc = pd.read_csv(hgnc_tsv, sep="\t", low_memory=False)

    # Pass 1 — current approved symbol (normalized key), first row wins.
    approved = {}
    for sym, hid, ent, ens in zip(
        hgnc["symbol"], hgnc["hgnc_id"], hgnc["entrez_id"], hgnc["ensembl_gene_id"]
    ):
        k = _norm(sym)
        if k is not None:
            approved.setdefault(k, (hid, ent, ens))

    # prev/alias maps: normalized alt symbol -> normalized approved-symbol key.
    prev_map, alias_map = {}, {}
    for col, store in [("prev_symbol", prev_map), ("alias_symbol", alias_map)]:
        for sym, val in zip(hgnc["symbol"], hgnc[col]):
            if pd.isna(val):
                continue
            ks = _norm(sym)
            for alt in str(val).split("|"):
                ka = _norm(alt)
                if ka and ka not in store:
                    store[ka] = ks

    rows = []
    for g in sorted(set(genes)):
        k = _norm(g)
        if k in approved:
            hid, ent, ens = approved[k]
        elif k in prev_map and prev_map[k] in approved:
            hid, ent, ens = approved[prev_map[k]]
        elif k in alias_map and alias_map[k] in approved:
            hid, ent, ens = approved[alias_map[k]]
        else:
            hid, ent, ens = (np.nan, np.nan, np.nan)
        rows.append((g, hid, ent, ens))

    return pd.DataFrame(
        rows, columns=["gene", "hgnc_id", "entrez_id", "ensembl_gene_id"]
    ).set_index("gene")
