#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
from typing import List

# =========================
# Configuration
# =========================
CONFIG = {
    "dataset_dir": "datasets",
    "feature_dir": "feature_output",
    "gtex_gct_gz": "../input_data/GTEx/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz",
    "pairs_xlsx": "biomarker_target_genes.xlsx",
    "gtex_chunksize": 1000,
}



# =========================
# Utilities & I/O (copy-adapted from 8_feature_extraction_expression_based_refractored.ipynb)
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_combined_clinical_main(dataset_dir: str, pairs_df: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for _, row in pairs_df.iterrows():
        sl_pair = f"{row['Biomarker']}_{row['Target']}"
        path = os.path.join(dataset_dir, f"PredictingSLResistanceFeatures_{sl_pair}_main.csv")
        frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def load_gtex_subset_for_genes(
    gtex_path: str,
    ensembl_ids: np.ndarray,
    chunksize: int,
) -> pd.DataFrame:
    keep = set([x for x in ensembl_ids if isinstance(x, str) and x.strip()])
    df_keep = pd.DataFrame()
    for chunk in pd.read_csv(gtex_path, sep="\t", skiprows=2, iterator=True, chunksize=chunksize):
        chunk_df = (
            chunk.assign(ensembl_id=chunk.Name.apply(lambda x: str(x).split(".")[0]))
                 .drop(columns=["Description"])
        )
        df_keep = pd.concat(
            [df_keep, chunk_df[chunk_df.ensembl_id.isin(keep)].set_index("ensembl_id")],
            axis=0,
        )
    return df_keep


def remove_duplicates_like_notebook(df_subset: pd.DataFrame) -> pd.DataFrame:
    df = df_subset.copy()
    duplicates = df[df.duplicated(subset=["ensembl_id"], keep=False)]
    non_zero_rows = duplicates[(duplicates.iloc[:, 2:] != 0).any(axis=1)]
    expr_data = df.drop(non_zero_rows.index)
    expr_data = expr_data.set_index("ensembl_id")
    expr_data.drop(columns=["Name"], inplace=True)
    return expr_data



# =========================
# Pure expression metrics
# =========================
def CalculateMeanExpression(expr_data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(expr_data.mean(axis=1), columns=["mean_expr"]).reset_index()


def CalculateVarianceExpression(expr_data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(expr_data.var(axis=1), columns=["expr_variance"]).reset_index()


def ExpandToPairs(mean_expr: pd.DataFrame, var_expr: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    pairs_expr = pd.merge(
        pairs,
        mean_expr.rename(columns={"ensembl_id": "A1_ensembl", "mean_expr": "A1_mean_expr"}),
        how="left",
    )
    pairs_expr = pd.merge(
        pairs_expr,
        mean_expr.rename(columns={"ensembl_id": "A2_ensembl", "mean_expr": "A2_mean_expr"}),
        how="left",
    )
    pairs_expr = pd.merge(
        pairs_expr,
        var_expr.rename(columns={"ensembl_id": "A1_ensembl", "expr_variance": "A1_expr_variance"}),
        how="left",
    )
    pairs_expr = pd.merge(
        pairs_expr,
        var_expr.rename(columns={"ensembl_id": "A2_ensembl", "expr_variance": "A2_expr_variance"}),
        how="left",
    )
    return pairs_expr


def CalculateExprCorrelation_Fast(pairs: pd.DataFrame, expr_data: pd.DataFrame) -> pd.DataFrame:
    mean_expr = CalculateMeanExpression(expr_data)
    var_expr = CalculateVarianceExpression(expr_data)
    pairs_expr = ExpandToPairs(mean_expr, var_expr, pairs)

    expr_ranks = expr_data.rank(axis=1)
    expr_dict = expr_ranks.to_dict(orient="index")

    def fast_spearman(pair):
        A1 = expr_dict.get(pair.A1_ensembl, None)
        A2 = expr_dict.get(pair.A2_ensembl, None)
        if A1 is None or A2 is None:
            return np.nan
        return np.corrcoef(list(A1.values()), list(A2.values()))[0, 1]

    pairs_expr["spearman_corr"] = pairs_expr.apply(fast_spearman, axis=1)
    return pairs_expr



# =========================
# Clinical orchestration: single GTEx load, per-partner correlation, per-pair split-write
# =========================
def build_pairs(combined: pd.DataFrame, partner: str) -> pd.DataFrame:
    assert partner in ("biomarker", "target1")
    col = f"ensembl_gene_id_{partner}"
    out = combined[["SL_Pair", col, "ensembl_gene_id_query"]].copy()
    out = out.rename(columns={col: "A1_ensembl", "ensembl_gene_id_query": "A2_ensembl"})
    return out.reset_index(drop=True)


def run_query_expression(
    combined: pd.DataFrame,
    partner: str,
    expr_data: pd.DataFrame,
    feature_dir: str,
) -> None:
    pairs = build_pairs(combined, partner)
    sl_pair_series = pairs["SL_Pair"].reset_index(drop=True)
    out = CalculateExprCorrelation_Fast(
        pairs.drop(columns=["SL_Pair"]),
        expr_data,
    ).reset_index(drop=True)
    out["SL_Pair"] = sl_pair_series

    for sl_pair, sub in out.groupby("SL_Pair", sort=False):
        name = f"gtex_co_expression_{partner}_query_{sl_pair}.csv"
        path = os.path.join(feature_dir, name)
        sub.drop(columns=["SL_Pair"]).to_csv(path, index=False)
        print(f"[OK] {path}  rows={len(sub):,}")



# =========================
# Run
# =========================
ensure_dir(CONFIG["feature_dir"])

pairs_df = pd.read_excel(CONFIG["pairs_xlsx"])
combined = load_combined_clinical_main(CONFIG["dataset_dir"], pairs_df)
print(f"[info] Combined clinical main: rows={len(combined):,}")

ensembl_union = pd.unique(
    combined[["ensembl_gene_id_biomarker", "ensembl_gene_id_target1", "ensembl_gene_id_query"]]
    .values.ravel()
)
print(f"[info] Unique ensembl IDs needed: {len(ensembl_union):,}")

df_genes = load_gtex_subset_for_genes(CONFIG["gtex_gct_gz"], ensembl_union, CONFIG["gtex_chunksize"])
df_genes_reset = df_genes.reset_index().rename(columns={"index": "ensembl_id"})
expr_data = remove_duplicates_like_notebook(df_genes_reset)
print(f"[info] GTEx expression matrix: {expr_data.shape}")

run_query_expression(combined, "biomarker", expr_data, CONFIG["feature_dir"])
run_query_expression(combined, "target1", expr_data, CONFIG["feature_dir"])