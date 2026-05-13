#!/usr/bin/env python
# coding: utf-8


import os
import re
import numpy as np
import pandas as pd


# ==========================================================
# A) CONFIG (keep names/paths exactly)
# ==========================================================
DEPMAP_GENE_EFFECT = "./input_data/DepMap/CRISPRGeneEffect.csv"
MAIN_DATASET_CSV   = "./input_data/3_ML_outputs/datasets/PredictingSLResistanceFeatures_main.csv"

FEATURE_OUT_DIR = "./input_data/3_ML_outputs/feature_output"

OUT_COESS_BIOMARKER = os.path.join(FEATURE_OUT_DIR, "coessentiality_biomarker_query.csv")
OUT_COESS_TARGET1   = os.path.join(FEATURE_OUT_DIR, "coessentiality_target1_query.csv")
OUT_COESS_TARGET2   = os.path.join(FEATURE_OUT_DIR, "coessentiality_target2_query.csv")

OUT_ESS_PERCENT = os.path.join(FEATURE_OUT_DIR, "essentiality_percentage_per_gene.csv")

ESSENTIALITY_THRESHOLD = -0.6  # same as notebook




# ==========================================================
# B) LOADERS (I/O only)
# ==========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_depmap_gene_effect(path: str = DEPMAP_GENE_EFFECT) -> pd.DataFrame:
    return pd.read_csv(path)


def load_main_dataset(path: str = MAIN_DATASET_CSV) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)




# ==========================================================
# C) PURE FUNCTIONS (no file I/O)
# ==========================================================
def extract_number(col_name: str) -> str:
    """
    Exact notebook logic:
      - extract digits inside parentheses: 'GENE (1234)' -> '1234'
      - if no match, keep original name
    """
    match = re.search(r"\((\d+)\)", str(col_name))
    return match.group(1) if match else str(col_name)


def preprocess_depmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Matches notebook:
      - rename 'Unnamed: 0' -> 'sample'
      - set index to 'sample'
      - drop 'sample'
      - rename columns using extract_number()
    """
    df = df.rename(columns={"Unnamed: 0": "sample"}).copy()
    df = df.set_index(df["sample"])
    df = df.drop(columns=["sample"])
    df.columns = [extract_number(c) for c in df.columns]
    return df


def compute_gene_effect_corr(depmap_matrix: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of genes (columns), same as notebook."""
    return depmap_matrix.corr(method="pearson")


def coessentiality_variance(depmap_matrix: pd.DataFrame) -> pd.Series:
    """Variance per gene (column), same as notebook: data.var(axis=0)."""
    return depmap_matrix.var(axis=0)


def essentiality_average(depmap_matrix: pd.DataFrame) -> pd.Series:
    """Mean per gene (column), same as notebook: data.mean(axis=0)."""
    return depmap_matrix.mean(axis=0)


def to_float_series(s: pd.Series) -> pd.Series:
    """Helper: cast IDs to float consistently with notebook behavior."""
    return s.astype(float)


def coessentiality_lookup(
    corr: pd.DataFrame,
    a: float,
    b: float
) -> float:
    """
    Same lookup logic:
      - if either missing -> 0.0
      - if not present in corr index/cols -> 0.0
      - else corr.loc[a, b]
    """
    if pd.isna(a) or pd.isna(b):
        return 0.0
    if (a in corr.index) and (b in corr.columns):
        return float(corr.loc[a, b])
    return 0.0


def essentiality_percentage_per_gene(depmap_matrix: pd.DataFrame, threshold: float = ESSENTIALITY_THRESHOLD) -> pd.DataFrame:
    """
    Exact notebook logic:
      - for each numeric column (gene), compute % of samples with value < threshold
      - denominator excludes NaN (notna().sum())
      - returns DataFrame(entrez_id, Essentiality_Percentage)
    """
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




# ==========================================================
# D) ORCHESTRATORS (ties everything together)
# ==========================================================
def build_coessentiality_biomarker_query(
    main_df: pd.DataFrame,
    corr: pd.DataFrame,
    var_s: pd.Series,
    avg_s: pd.Series
) -> pd.DataFrame:
    """
    Matches notebook cell 6 output exactly (columns & logic):
      - uses entrez_id_query + entrez_id_biomarker
      - Correlation (0.0 fallback)
      - Biomarker_Essentiality_Variance uses query gene variance
      - Biomarker_Essentiality_Average uses query gene mean then * -1
    """
    out = main_df[["entrez_id_query", "entrez_id_biomarker"]].copy()

    out["entrez_id_query"] = to_float_series(out["entrez_id_query"])
    out["entrez_id_biomarker"] = to_float_series(out["entrez_id_biomarker"])

    def corr_row(r):
        return coessentiality_lookup(corr, r["entrez_id_query"], r["entrez_id_biomarker"])

    def query_var(r):
        gid = r["entrez_id_query"]
        if pd.isna(gid):
            return np.nan
        return var_s.get(gid, np.nan)

    def query_avg(r):
        gid = r["entrez_id_query"]
        if pd.isna(gid):
            return np.nan
        return avg_s.get(gid, np.nan)

    out["Correlation"] = out.apply(corr_row, axis=1)
    out["Biomarker_Essentiality_Variance"] = out.apply(query_var, axis=1)
    out["Biomarker_Essentiality_Average"] = out.apply(query_avg, axis=1)
    out["Biomarker_Essentiality_Average"] = out["Biomarker_Essentiality_Average"] * -1

    return out


def build_coessentiality_target_query(
    main_df: pd.DataFrame,
    corr: pd.DataFrame,
    target_col: str
) -> pd.DataFrame:
    """
    Matches notebook cells 7/8 outputs:
      - uses entrez_id_query + entrez_id_targetX
      - Correlation (0.0 fallback)
    """
    out = main_df[["entrez_id_query", target_col]].copy()
    out["entrez_id_query"] = to_float_series(out["entrez_id_query"])
    out[target_col] = to_float_series(out[target_col])

    def corr_row(r):
        return coessentiality_lookup(corr, r["entrez_id_query"], r[target_col])

    out["Correlation"] = out.apply(corr_row, axis=1)
    return out




# ==========================================================
# E) RUN
# ==========================================================
def main() -> None:
    ensure_dir(FEATURE_OUT_DIR)

    # 1) DepMap preprocessing + correlation
    depmap_raw = load_depmap_gene_effect(DEPMAP_GENE_EFFECT)
    depmap = preprocess_depmap(depmap_raw)

    # Notebook does: data.columns = data.columns.astype(float)
    # We'll match that (and keep consistent with corr index/columns float)
    depmap.columns = depmap.columns.astype(float)

    corr = compute_gene_effect_corr(depmap)
    corr.index = corr.index.astype(float)
    corr.columns = corr.columns.astype(float)

    var_s = coessentiality_variance(depmap)   # index: float entrez
    avg_s = essentiality_average(depmap)      # index: float entrez

    # 2) Main ML dataset
    main_df = load_main_dataset(MAIN_DATASET_CSV)

    # 3) Outputs: biomarker/query
    out_bio = build_coessentiality_biomarker_query(main_df, corr, var_s, avg_s)
    out_bio.to_csv(OUT_COESS_BIOMARKER, index=False)

    # 4) Outputs: target1/query and target2/query
    out_t1 = build_coessentiality_target_query(main_df, corr, "entrez_id_target1")
    out_t1.to_csv(OUT_COESS_TARGET1, index=False)

    out_t2 = build_coessentiality_target_query(main_df, corr, "entrez_id_target2")
    out_t2.to_csv(OUT_COESS_TARGET2, index=False)

    # 5) Essentiality percentage per gene
    ess_pct = essentiality_percentage_per_gene(depmap, threshold=ESSENTIALITY_THRESHOLD)
    ess_pct.to_csv(OUT_ESS_PERCENT, index=False)

    print("[OK] Wrote:")
    print(" -", OUT_COESS_BIOMARKER)
    print(" -", OUT_COESS_TARGET1)
    print(" -", OUT_COESS_TARGET2)
    print(" -", OUT_ESS_PERCENT)


main()