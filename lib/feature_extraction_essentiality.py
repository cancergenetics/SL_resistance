"""Shared library — DepMap essentiality feature extraction.

Outputs (in feature_output_dir):
  coessentiality_biomarker_query.csv  — gene-effect Pearson corr (query × biomarker) + variance + mean
  coessentiality_target1_query.csv    — gene-effect Pearson corr (query × target1)
  coessentiality_target2_query.csv    — gene-effect Pearson corr (query × target2)
  essentiality_percentage_per_gene.csv — % DepMap samples with gene-effect < -0.6

Public entry point: extract_essentiality(features_main_schema_csv, feature_output_dir, root)
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
    return s.astype(float)


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


def extract_essentiality(features_main_schema_csv: Path, feature_output_dir: Path,
                         root: Path) -> None:
    features_main_schema_csv = Path(features_main_schema_csv)
    feature_output_dir = Path(feature_output_dir)
    root = Path(root)

    DEPMAP_GENE_EFFECT = root / "input_data" / "DepMap" / "CRISPRGeneEffect.csv"
    OUT_COESS_BIOMARKER = feature_output_dir / "coessentiality_biomarker_query.csv"
    OUT_COESS_TARGET1 = feature_output_dir / "coessentiality_target1_query.csv"
    OUT_COESS_TARGET2 = feature_output_dir / "coessentiality_target2_query.csv"
    OUT_ESS_PERCENT = feature_output_dir / "essentiality_percentage_per_gene.csv"

    feature_output_dir.mkdir(parents=True, exist_ok=True)

    depmap_raw = pd.read_csv(DEPMAP_GENE_EFFECT)
    depmap = preprocess_depmap(depmap_raw)
    depmap.columns = depmap.columns.astype(float)

    corr = compute_gene_effect_corr(depmap)
    corr.index = corr.index.astype(float)
    corr.columns = corr.columns.astype(float)

    var_s = coessentiality_variance(depmap)
    avg_s = essentiality_average(depmap)

    main_df = pd.read_csv(features_main_schema_csv, low_memory=False)

    build_coessentiality_biomarker_query(main_df, corr, var_s, avg_s).to_csv(OUT_COESS_BIOMARKER, index=False)
    build_coessentiality_target_query(main_df, corr, "entrez_id_target1").to_csv(OUT_COESS_TARGET1, index=False)
    build_coessentiality_target_query(main_df, corr, "entrez_id_target2").to_csv(OUT_COESS_TARGET2, index=False)
    essentiality_percentage_per_gene(depmap, ESSENTIALITY_THRESHOLD).to_csv(OUT_ESS_PERCENT, index=False)

    print("[OK] Wrote:")
    for p in (OUT_COESS_BIOMARKER, OUT_COESS_TARGET1, OUT_COESS_TARGET2, OUT_ESS_PERCENT):
        print(" -", p)
