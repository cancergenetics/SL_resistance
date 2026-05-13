"""Shared library — GTEx co-expression feature extraction.

For each (query, biomarker) and (query, target1) pair, compute:
  - A1/A2 mean expression across GTEx tissues
  - A1/A2 variance across tissues
  - Spearman correlation across tissues (rank-based Pearson)

Public entry point: extract_expression(features_main_schema_csv, feature_output_dir, root)
"""

from __future__ import annotations

from pathlib import Path

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


def build_pairs(main_df: pd.DataFrame, a_col: str, b_col: str) -> pd.DataFrame:
    return main_df[[a_col, b_col]].rename(columns={a_col: "A1_ensembl", b_col: "A2_ensembl"})


def run_one_coexpression(main_df: pd.DataFrame, a_col: str, b_col: str,
                         out_csv: Path, gtex_gct_gz: Path) -> pd.DataFrame:
    pairs = build_pairs(main_df, a_col, b_col)
    genes = pd.unique(pairs[["A1_ensembl", "A2_ensembl"]].values.ravel())
    df_genes = load_gtex_subset_for_genes(gtex_gct_gz, genes, chunksize=GTEX_CHUNKSIZE)
    df_genes_reset = df_genes.reset_index().rename(columns={"index": "ensembl_id"})
    expr_data = remove_duplicates_like_notebook(df_genes_reset)
    out = CalculateExprCorrelation_Fast(pairs, expr_data)
    out.to_csv(out_csv, index=False)
    return out


def extract_expression(features_main_schema_csv: Path, feature_output_dir: Path,
                       root: Path) -> None:
    features_main_schema_csv = Path(features_main_schema_csv)
    feature_output_dir = Path(feature_output_dir)
    root = Path(root)

    GTEX_GCT_GZ = root / "input_data" / "GTEx" / "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"
    OUT_BIOMARKER = feature_output_dir / "gtex_co_expression_biomarker_query.csv"
    OUT_TARGET1 = feature_output_dir / "gtex_co_expression_target1_query.csv"

    feature_output_dir.mkdir(parents=True, exist_ok=True)
    main_df = pd.read_csv(features_main_schema_csv, low_memory=False)
    run_one_coexpression(main_df, "ensembl_gene_id_biomarker", "ensembl_gene_id_query",
                         OUT_BIOMARKER, GTEX_GCT_GZ)
    run_one_coexpression(main_df, "ensembl_gene_id_target1", "ensembl_gene_id_query",
                         OUT_TARGET1, GTEX_GCT_GZ)
    print("[OK] Wrote:\n -", OUT_BIOMARKER, "\n -", OUT_TARGET1)
