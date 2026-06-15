#!/usr/bin/env python
# coding: utf-8


import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

# =========================
# Configuration
# =========================
CONFIG = {
    "pairs_xlsx": "biomarker_target_genes.xlsx",
    "hgnc_file": "../input_data/HGNC/hgnc_complete_set.txt",
    "dataset_dir": "datasets",
    "screen_label": "Clinical_Trials",
}

FEATURE_PLACEHOLDER_COLS = [
    "StringInteractionWithBiomarker",
    "StringInteractionWithTarget",
    "CoexpressionWithBiomarker",
    "CoexpressionWithTarget",
    "AvgExpression",
    "CoessentialityWithBiomarker",
    "CoessentialityWithTarget",
    "FET_SharedInteractors_Biomarker_BIOGRID",
    "FET_SharedInteractors_Target_BIOGRID",
    "FET_SharedInteractors_Biomarker_STRING",
    "FET_SharedInteractors_Target_STRING",
]



# =========================
# Utilities
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_symbol(s: Optional[str]) -> Optional[str]:
    return s.strip().upper() if isinstance(s, str) and s.strip() else None


def build_gene_lookup(hgnc_file: str) -> Dict[str, Dict[str, Optional[str]]]:
    """Symbol -> {hgnc_id, ensembl_gene_id, entrez_id} lookup (first-win over aliases)."""
    hgnc = pd.read_csv(hgnc_file, sep="\t", low_memory=False)

    cols_needed = [
        "hgnc_id", "symbol", "prev_symbol", "alias_symbol",
        "ensembl_gene_id", "entrez_id",
    ]
    missing = [c for c in cols_needed if c not in hgnc.columns]
    if missing:
        raise ValueError(f"HGNC file missing columns: {missing}")

    def split_multi(v) -> List[str]:
        if not isinstance(v, str) or not v.strip():
            return []
        return [x.strip() for x in re.split(r"[|,;]", v) if x.strip()]

    lookup: Dict[str, Dict[str, Optional[str]]] = {}

    # Precompute (canonical symbol, prev/alias symbols, id bundle) per HGNC row
    rows = []
    for _, row in hgnc[cols_needed].iterrows():
        bundle = {
            "hgnc_id": row.get("hgnc_id", np.nan),
            "ensembl_gene_id": row.get("ensembl_gene_id", np.nan),
            "entrez_id": row.get("entrez_id", np.nan),
        }
        bundle = {k: (None if pd.isna(v) else str(v)) for k, v in bundle.items()}
        canonical = normalize_symbol(row.get("symbol"))
        prev = [s for s in (normalize_symbol(x) for x in split_multi(row.get("prev_symbol"))) if s]
        alias = [s for s in (normalize_symbol(x) for x in split_multi(row.get("alias_symbol"))) if s]
        rows.append((canonical, prev, alias, bundle))

    # Strict 3-tier precedence: current approved symbol > prev_symbol (former approved
    # name) > alias_symbol. Canonical never shadowed by another gene's prev/alias
    # ("ATR" -> ATR kinase, not ANTXR1 alias); a symbol that is a prev of one gene and an
    # alias of another resolves to the prev gene ("MLL2" = prev of KMT2D, alias of KMT2B
    # -> KMT2D). Within a tier, first HGNC row wins.
    for canonical, _prev, _alias, bundle in rows:
        if canonical:
            lookup.setdefault(canonical, bundle)
    for _canonical, prev, _alias, bundle in rows:
        for ns in prev:
            lookup.setdefault(ns, bundle)
    for _canonical, _prev, alias, bundle in rows:
        for ns in alias:
            lookup.setdefault(ns, bundle)
    return lookup


def lookup_ids(
    gene_lookup: Dict[str, Dict[str, Optional[str]]],
    symbol: Optional[str],
) -> Dict[str, Optional[str]]:
    sym = normalize_symbol(symbol)
    if not sym:
        return {"hgnc_id": None, "ensembl_gene_id": None, "entrez_id": None}
    return gene_lookup.get(sym, {"hgnc_id": None, "ensembl_gene_id": None, "entrez_id": None})


def load_protein_coding_genes(hgnc_file: str) -> pd.DataFrame:
    """Return HGNC rows restricted to approved protein-coding genes."""
    hgnc = pd.read_csv(hgnc_file, sep="\t", low_memory=False)
    mask = (hgnc["locus_group"] == "protein-coding gene") & (hgnc["status"] == "Approved")
    sub = hgnc.loc[mask, ["symbol", "hgnc_id", "ensembl_gene_id", "entrez_id"]].copy()
    sub = sub.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
    return sub.reset_index(drop=True)



# =========================
# SL-pair parsing & ML-table transforms
# =========================
def split_sl_pair(sl_pair: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Clinical pairs are Biomarker_Target (single target); Target2 is None."""
    if not isinstance(sl_pair, str) or not sl_pair.strip():
        return None, None, None
    parts = [p.strip() for p in sl_pair.split("_") if p.strip()]
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return parts[0], (parts[1] if len(parts) > 1 else None), None


def add_biomarker_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = df["SL_Pair"].apply(split_sl_pair)
    df["Biomarker"] = parsed.apply(lambda x: x[0])
    df["Target1"] = parsed.apply(lambda x: x[1])
    df["Target2"] = parsed.apply(lambda x: x[2])
    return df


def add_id_columns_from_lookup(
    df: pd.DataFrame,
    gene_lookup: Dict[str, Dict[str, Optional[str]]],
) -> pd.DataFrame:
    df = df.copy()
    for role, gene_col in [("biomarker", "Biomarker"), ("target1", "Target1"), ("target2", "Target2")]:
        ids = df[gene_col].apply(lambda s: lookup_ids(gene_lookup, s))
        df[f"entrez_id_{role}"] = ids.apply(lambda d: d["entrez_id"])
        df[f"hgnc_id_{role}"] = ids.apply(lambda d: d["hgnc_id"])
        df[f"ensembl_gene_id_{role}"] = ids.apply(lambda d: d["ensembl_gene_id"])
    return df


def add_feature_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in FEATURE_PLACEHOLDER_COLS:
        if c not in df.columns:
            df[c] = None
    return df


def reorder_columns_like_main(df: pd.DataFrame) -> pd.DataFrame:
    desired_front = [
        "Screen", "SL_Pair",
        "Query", "Biomarker", "Target1", "Target2",
        "entrez_id_query", "hgnc_id_query", "ensembl_gene_id_query",
        "entrez_id_biomarker", "hgnc_id_biomarker", "ensembl_gene_id_biomarker",
        "entrez_id_target1", "hgnc_id_target1", "ensembl_gene_id_target1",
        "entrez_id_target2", "hgnc_id_target2", "ensembl_gene_id_target2",
    ]
    front = [c for c in desired_front if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]



# =========================
# Clinical orchestration
# =========================
def build_clinical_main_dataset(
    pairs_df: pd.DataFrame,
    protein_coding: pd.DataFrame,
    gene_lookup: Dict[str, Dict[str, Optional[str]]],
    screen_label: str = "Clinical_Trials",
) -> pd.DataFrame:
    """
    Build a single combined ML dataset spanning all clinical SL pairs.

    For each (Biomarker, Target) row in pairs_df, stamps the full protein-coding gene
    universe (including the biomarker and target themselves) with SL_Pair / Biomarker /
    Target1 / Target2=None.
    Concatenates into one dataframe, then runs Biomarker/Target column derivation,
    HGNC ID lookup, and placeholder-feature insertion once across the whole frame.
    """
    base = protein_coding.rename(columns={
        "symbol": "Query",
        "hgnc_id": "hgnc_id_query",
        "ensembl_gene_id": "ensembl_gene_id_query",
        "entrez_id": "entrez_id_query",
    }).copy()

    per_pair_frames: List[pd.DataFrame] = []
    for _, row in pairs_df.iterrows():
        biomarker = str(row["Biomarker"]).strip()
        target1 = str(row["Target"]).strip()
        sl_pair = f"{biomarker}_{target1}"

        # Include the full protein-coding universe, INCLUDING the biomarker and target
        # themselves: a real CRISPR screen library cannot drop the target/biomarker, so
        # they appear as query genes in the screens (and in validation, e.g. Knoll).
        sub = base.copy()
        sub.insert(0, "Screen", screen_label)
        sub.insert(1, "SL_Pair", sl_pair)
        per_pair_frames.append(sub)

    combined = pd.concat(per_pair_frames, ignore_index=True)
    combined = add_biomarker_target_columns(combined)
    combined = add_id_columns_from_lookup(combined, gene_lookup)
    combined = add_feature_placeholders(combined)
    combined = reorder_columns_like_main(combined)
    return combined


def write_per_pair_datasets(combined: pd.DataFrame, dataset_dir: str) -> None:
    ensure_dir(dataset_dir)
    for sl_pair, sub in combined.groupby("SL_Pair", sort=False):
        out_path = os.path.join(dataset_dir, f"PredictingSLResistanceFeatures_{sl_pair}_main.csv")
        sub.to_csv(out_path, index=False)
        print(f"[OK] {out_path}  rows={len(sub):,}")



# =========================
# Run
# =========================
pairs_df = pd.read_excel(CONFIG["pairs_xlsx"])
print(f"[info] Loaded {len(pairs_df)} biomarker-target pairs from {CONFIG['pairs_xlsx']}")

protein_coding = load_protein_coding_genes(CONFIG["hgnc_file"])
print(f"[info] Protein-coding gene universe: {len(protein_coding):,} genes")

gene_lookup = build_gene_lookup(CONFIG["hgnc_file"])
print(f"[info] HGNC symbol lookup: {len(gene_lookup):,} aliases")

clinical_main_df = build_clinical_main_dataset(
    pairs_df=pairs_df,
    protein_coding=protein_coding,
    gene_lookup=gene_lookup,
    screen_label=CONFIG["screen_label"],
)
print(f"[info] Combined clinical main: rows={len(clinical_main_df):,}  cols={clinical_main_df.shape[1]}")

write_per_pair_datasets(clinical_main_df, CONFIG["dataset_dir"])
clinical_main_df.head()