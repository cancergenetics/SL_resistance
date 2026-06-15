"""Shared library — GTEx co-expression feature extraction CORE (single source of truth).

Holds the GTEx loading + co-expression computation building blocks shared by the
main pipeline (`08_feature_extraction_expression_based.py`) and the clinical-trials
pipeline (`clinical_trials/03_feature_extraction_expression_based.py`). Each script
keeps only its own I/O orchestration (the main pipeline loads GTEx per partner and
writes one combined table per partner; clinical loads GTEx once over the union and
writes per-SL_pair files) and imports the functions below.

Everything keys on Ensembl gene IDs taken from the main dataset (already resolved
via the shared HGNC resolver in `06`), so there is no HGNC lookup here.

For each (A1, A2) gene pair, computes A1/A2 mean + variance expression across GTEx
tissues and the Spearman correlation (rank-based Pearson) across tissues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GTEX_CHUNKSIZE = 1000


def load_gtex_subset_for_genes(gtex_path, ensembl_ids, chunksize: int = GTEX_CHUNKSIZE) -> pd.DataFrame:
    keep = set([x for x in ensembl_ids if isinstance(x, str) and x.strip()])
    df_keep = pd.DataFrame()
    for chunk in pd.read_csv(gtex_path, sep="\t", skiprows=2, iterator=True, chunksize=chunksize):
        chunk_df = (
            chunk.assign(ensembl_id=chunk.Name.apply(lambda x: str(x).split(".")[0]))
            .drop(columns=["Description"])
        )
        df_keep = pd.concat(
            [df_keep, chunk_df[chunk_df.ensembl_id.isin(keep)].set_index("ensembl_id")], axis=0
        )
    return df_keep


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
    print(
        "Pairs where both genes have expr:",
        pairs_expr[(~pairs_expr.A1_mean_expr.isna()) & (~pairs_expr.A2_mean_expr.isna())].shape[0],
        "/", pairs_expr.shape[0],
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


def remove_duplicates_like_notebook(df_subset: pd.DataFrame) -> pd.DataFrame:
    df = df_subset.copy()
    duplicates = df[df.duplicated(subset=["ensembl_id"], keep=False)]
    non_zero_rows = duplicates[(duplicates.iloc[:, 2:] != 0).any(axis=1)]
    expr_data = df.drop(non_zero_rows.index)
    expr_data = expr_data.set_index("ensembl_id")
    expr_data.drop(columns=["Name"], inplace=True)
    return expr_data
