"""Shared library — DepMap essentiality feature extraction CORE (single source of truth).

Holds the DepMap loading + co-essentiality computation building blocks shared by the
main pipeline (`09_feature_extraction_essentiality_based.py`) and the clinical-trials
pipeline (`clinical_trials/04_feature_extraction_essentiality_based.py`). Each script
keeps only its own I/O orchestration (the main pipeline writes one combined table per
partner; clinical writes per-SL_pair files) and imports the functions below.

Everything keys on Entrez gene IDs taken from the main dataset (already resolved via
the shared HGNC resolver in `06`), so there is no HGNC lookup here.

Provides: extract_number, preprocess_depmap, compute_gene_effect_corr,
coessentiality_variance, essentiality_average, coessentiality_lookup, to_float_series,
essentiality_percentage_per_gene, build_coessentiality_biomarker_query,
build_coessentiality_target_query.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ESSENTIALITY_THRESHOLD = -0.6


def extract_number(col_name: str) -> str:
    match = re.search(r"\((\d+)\)", str(col_name))
    return match.group(1) if match else str(col_name)


def preprocess_depmap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Unnamed: 0": "sample"}).copy()
    df = df.set_index(df["sample"]).drop(columns=["sample"])
    df.columns = [extract_number(c) for c in df.columns]
    return df


def compute_gene_effect_corr(depmap_matrix: pd.DataFrame) -> pd.DataFrame:
    return depmap_matrix.corr(method="pearson")


def coessentiality_variance(depmap_matrix: pd.DataFrame) -> pd.Series:
    return depmap_matrix.var(axis=0)


def essentiality_average(depmap_matrix: pd.DataFrame) -> pd.Series:
    return depmap_matrix.mean(axis=0)


def to_float_series(s: pd.Series) -> pd.Series:
    # coerce so stray non-numeric / "NA" -> NaN instead of raising
    return pd.to_numeric(s, errors="coerce").astype(float)


def coessentiality_lookup(corr: pd.DataFrame, a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return 0.0
    if (a in corr.index) and (b in corr.columns):
        return float(corr.loc[a, b])
    return 0.0


def essentiality_percentage_per_gene(depmap_matrix: pd.DataFrame,
                                     threshold: float = ESSENTIALITY_THRESHOLD) -> pd.DataFrame:
    result = {}
    for col in depmap_matrix.columns:
        if pd.api.types.is_numeric_dtype(depmap_matrix[col]):
            total = depmap_matrix[col].notna().sum()
            if total > 0:
                count = (depmap_matrix[col] < threshold).sum()
                result[col] = (count / total) * 100
            else:
                result[col] = None
    return pd.DataFrame(list(result.items()), columns=["entrez_id", "Essentiality_Percentage"])


def build_coessentiality_biomarker_query(main_df: pd.DataFrame, corr: pd.DataFrame,
                                         var_s: pd.Series, avg_s: pd.Series) -> pd.DataFrame:
    out = main_df[["entrez_id_query", "entrez_id_biomarker"]].copy()
    out["entrez_id_query"] = to_float_series(out["entrez_id_query"])
    out["entrez_id_biomarker"] = to_float_series(out["entrez_id_biomarker"])

    out["Correlation"] = out.apply(
        lambda r: coessentiality_lookup(corr, r["entrez_id_query"], r["entrez_id_biomarker"]), axis=1
    )
    out["Biomarker_Essentiality_Variance"] = out["entrez_id_query"].apply(
        lambda gid: np.nan if pd.isna(gid) else var_s.get(gid, np.nan)
    )
    out["Biomarker_Essentiality_Average"] = out["entrez_id_query"].apply(
        lambda gid: np.nan if pd.isna(gid) else avg_s.get(gid, np.nan)
    ) * -1
    return out


def build_coessentiality_target_query(main_df: pd.DataFrame, corr: pd.DataFrame,
                                      target_col: str) -> pd.DataFrame:
    out = main_df[["entrez_id_query", target_col]].copy()
    out["entrez_id_query"] = to_float_series(out["entrez_id_query"])
    out[target_col] = to_float_series(out[target_col])
    out["Correlation"] = out.apply(
        lambda r: coessentiality_lookup(corr, r["entrez_id_query"], r[target_col]), axis=1
    )
    return out
