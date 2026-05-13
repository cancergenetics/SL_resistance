#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd


# ==========================================================
# A) CONFIG  (paths/names EXACTLY as your notebook)
# ==========================================================
MAIN_DATASET_CSV = "./input_data/3_ML_outputs/datasets/PredictingSLResistanceFeatures_main.csv"
GTEX_GCT_GZ = "./input_data/GTEx/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"

FEATURE_OUT_DIR = "./input_data/3_ML_outputs/feature_output"
OUT_BIOMARKER = os.path.join(FEATURE_OUT_DIR, "gtex_co_expression_biomarker_query.csv")
OUT_TARGET1   = os.path.join(FEATURE_OUT_DIR, "gtex_co_expression_target1_query.csv")
OUT_TARGET2   = os.path.join(FEATURE_OUT_DIR, "gtex_co_expression_target2_query.csv")

GTEX_CHUNKSIZE = 1000




# ==========================================================
# B) LOADERS (I/O only)
# ==========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_main_dataset(path: str = MAIN_DATASET_CSV) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_gtex_subset_for_genes(
    gtex_path: str,
    ensembl_ids: np.ndarray,
    chunksize: int = GTEX_CHUNKSIZE
) -> pd.DataFrame:
    """
    Same approach as original:
      - read GTEx .gct.gz in chunks
      - skiprows=2
      - ensembl_id = Name split('.')[0]
      - drop Description
      - keep only requested genes
      - set_index('ensembl_id') per chunk (same net effect)
    """
    keep = set([x for x in ensembl_ids if isinstance(x, str) and x.strip()])

    df_keep = pd.DataFrame()
    for chunk in pd.read_csv(gtex_path, sep="\t", skiprows=2, iterator=True, chunksize=chunksize):
        chunk_df = (
            chunk.assign(ensembl_id=chunk.Name.apply(lambda x: str(x).split(".")[0]))
                 .drop(columns=["Description"])
        )
        df_keep = pd.concat(
            [df_keep, chunk_df[chunk_df.ensembl_id.isin(keep)].set_index("ensembl_id")],
            axis=0
        )

    return df_keep




# ==========================================================
# C) PURE FUNCTIONS (no file I/O)
# ==========================================================
def CalculateMeanExpression(expr_data: pd.DataFrame) -> pd.DataFrame:
    # same as notebook: mean across tissues per gene
    mean_expr = pd.DataFrame(expr_data.mean(axis=1), columns=["mean_expr"]).reset_index()
    return mean_expr


def CalculateVarianceExpression(expr_data: pd.DataFrame) -> pd.DataFrame:
    # ✅ missing in refactor before; same as notebook: variance across tissues per gene
    var_expr = pd.DataFrame(expr_data.var(axis=1), columns=["expr_variance"]).reset_index()
    return var_expr


def ExpandToPairs(mean_expr: pd.DataFrame, var_expr: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Matches the *final* logic in your original notebook:
      - merges mean expression for A1/A2
      - merges variance expression for A1/A2
      - prints the same diagnostics
      - (original creates subset for printing but returns full pairs_expr)
    """
    # Add mean expression
    pairs_expr = pd.merge(
        pairs,
        mean_expr.rename(columns={"ensembl_id": "A1_ensembl", "mean_expr": "A1_mean_expr"}),
        how="left"
    )
    pairs_expr = pd.merge(
        pairs_expr,
        mean_expr.rename(columns={"ensembl_id": "A2_ensembl", "mean_expr": "A2_mean_expr"}),
        how="left"
    )

    # Add variance expression
    pairs_expr = pd.merge(
        pairs_expr,
        var_expr.rename(columns={"ensembl_id": "A1_ensembl", "expr_variance": "A1_expr_variance"}),
        how="left"
    )
    pairs_expr = pd.merge(
        pairs_expr,
        var_expr.rename(columns={"ensembl_id": "A2_ensembl", "expr_variance": "A2_expr_variance"}),
        how="left"
    )

    print(
        "Pairs where both genes have expr:",
        pairs_expr[(~pairs_expr.A1_mean_expr.isna()) & (~pairs_expr.A2_mean_expr.isna())].shape[0],
        "/",
        pairs_expr.shape[0]
    )

    # subset only for printing (same as notebook)
    pairs_expr_subset = pairs_expr[
        (~pairs_expr.A1_mean_expr.isna()) & (~pairs_expr.A2_mean_expr.isna()) &
        (pairs_expr.A1_mean_expr != 0) & (pairs_expr.A2_mean_expr != 0)
    ]
    print("# Pairs where both have expression:", pairs_expr_subset.shape[0])
    print("# Total Pairs:", pairs.shape[0])

    # IMPORTANT: return full table (same as your notebook)
    return pairs_expr


def CalculateExprCorrelation_Fast(pairs: pd.DataFrame, expr_data: pd.DataFrame) -> pd.DataFrame:
    """
    Matches the *final* CalculateExprCorrelation_Fast in your notebook:
      - mean + variance expansion
      - Spearman = Pearson(corrcoef) on ranks
      - missing genes -> NaN
      - returns expanded pairs table + spearman_corr
    """
    mean_expr = CalculateMeanExpression(expr_data)
    var_expr = CalculateVarianceExpression(expr_data)

    pairs_expr = ExpandToPairs(mean_expr, var_expr, pairs)

    expr_ranks = expr_data.rank(axis=1)  # rank across tissues per gene
    expr_dict = expr_ranks.to_dict(orient="index")

    def fast_spearman(pair):
        A1 = expr_dict.get(pair.A1_ensembl, None)
        A2 = expr_dict.get(pair.A2_ensembl, None)
        if A1 is None or A2 is None:
            return np.nan
        return np.corrcoef(list(A1.values()), list(A2.values()))[0, 1]

    pairs_expr["spearman_corr"] = pairs_expr.apply(fast_spearman, axis=1)
    return pairs_expr


def remove_duplicates_like_notebook(df_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Exact duplicate handling from your notebook.
    """
    df = df_subset.copy()

    duplicates = df[df.duplicated(subset=["ensembl_id"], keep=False)]
    non_zero_rows = duplicates[(duplicates.iloc[:, 2:] != 0).any(axis=1)]
    expr_data = df.drop(non_zero_rows.index)

    expr_data = expr_data.set_index("ensembl_id")
    expr_data.drop(columns=["Name"], inplace=True)
    return expr_data


def build_pairs(main_df: pd.DataFrame, a_col: str, b_col: str) -> pd.DataFrame:
    pairs = main_df[[a_col, b_col]].copy()
    pairs = pairs.rename(columns={a_col: "A1_ensembl", b_col: "A2_ensembl"})
    return pairs




# ==========================================================
# D) ORCHESTRATION
# ==========================================================
def run_one_coexpression(main_df: pd.DataFrame, a_col: str, b_col: str, out_csv: str) -> pd.DataFrame:
    pairs = build_pairs(main_df, a_col, b_col)
    genes = pd.unique(pairs[["A1_ensembl", "A2_ensembl"]].values.ravel())

    df_genes = load_gtex_subset_for_genes(GTEX_GCT_GZ, genes, chunksize=GTEX_CHUNKSIZE)

    # restore ensembl_id column for duplicate logic
    df_genes_reset = df_genes.reset_index().rename(columns={"index": "ensembl_id"})
    expr_data = remove_duplicates_like_notebook(df_genes_reset)

    out = CalculateExprCorrelation_Fast(pairs, expr_data)
    out.to_csv(out_csv, index=False)
    return out


def main() -> None:
    ensure_dir(FEATURE_OUT_DIR)

    main_df = load_main_dataset(MAIN_DATASET_CSV)

    run_one_coexpression(main_df, "ensembl_gene_id_biomarker", "ensembl_gene_id_query", OUT_BIOMARKER)
    run_one_coexpression(main_df, "ensembl_gene_id_target1",   "ensembl_gene_id_query", OUT_TARGET1)
    run_one_coexpression(main_df, "ensembl_gene_id_target2",   "ensembl_gene_id_query", OUT_TARGET2)

    print("[OK] Wrote:")
    print(" -", OUT_BIOMARKER)
    print(" -", OUT_TARGET1)
    print(" -", OUT_TARGET2)



# ==========================================================
# E) RUN
# ==========================================================
main()