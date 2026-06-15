#!/usr/bin/env python
# coding: utf-8


import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

# Single shared HGNC resolver (strict 3-tier symbol > prev > alias, case-insensitive).
from lib.hgnc_lookup import build_hgnc_lookup


# =========================
# Configuration
# =========================
INPUT_SCREENS_DIR = "./input_data/2_outputs_with_hgnc"
HGNC_FILE = "./input_data/HGNC/hgnc_complete_set.txt"

ANALYSIS_PAIR_STRING = "./input_data/2_outputs_with_hgnc/analysis_pair_string.csv"
ANALYSIS_PAIR_BIOGRID = "./input_data/2_outputs_with_hgnc/analysis_pair_biogrid.csv"

OUTPUT_DATASET_DIR = "./input_data/3_ML_outputs/datasets"
OUTPUT_MAIN_FILE = os.path.join(OUTPUT_DATASET_DIR, "PredictingSLResistanceFeatures_main.csv")

# The columns you want as blank placeholders in the final ML table
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


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Heuristically detect key columns in analysis_pair_*.csv without changing your file formats.
    Returns detected column names for:
      - screen
      - sl_pair
      - aggregated
    """
    cols = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    screen_col = pick("screen_prefix", "screen", "screen_name", "screenprefix")
    pair_col = pick("sl_pair", "slpair", "pair_name", "pair", "slpair_name")

    # aggregated flag can appear in many forms
    agg_col = pick(
        "aggregated", "is_aggregated", "is_agg", "agg", "aggregation", "isaggregated"
    )

    return {"screen": screen_col, "sl_pair": pair_col, "aggregated": agg_col}


def parse_aggregated_value(x) -> bool:
    """
    Convert various representations to boolean aggregated/non-aggregated.
    Accepts: True/False, 1/0, 'yes/no', 'true/false', 'aggregated', etc.
    """
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "agg", "aggregated"}




# =========================
# Gene lookup (HGNC) — delegated to the shared resolver lib.hgnc_lookup
# (strict 3-tier symbol > prev_symbol > alias, case-insensitive). Built per-run
# over the biomarker/target symbols inside add_id_columns_from_lookup().
# =========================



# =========================
# SL-pair parsing
# =========================
def split_sl_pair(sl_pair: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Mirrors your original logic, but made safer. Returns (Biomarker, Target1, Target2, Target3).

    - Any pair containing AKT -> (GeneA, AKT1, AKT2, AKT3)   # pan-AKT inhibitor: 3 targets
    - Any pair containing MEK -> (GeneA, MEK1, MEK2, None)
    - CDK4_6 special handling -> (GeneA, CDK4, CDK6, None)
    - Standard: 2 genes -> (A, B, None, None)
    - Standard: 3 parts -> (A, B, C, None)
    """
    if not isinstance(sl_pair, str) or not sl_pair.strip():
        return None, None, None, None

    parts = [p.strip() for p in sl_pair.split("_") if p.strip()]

    # Pan-AKT inhibitor (e.g. PTEN_AKT) targets all three AKT isoforms
    if "AKT" in parts:
        return parts[0], "AKT1", "AKT2", "AKT3"

    if "MEK" in parts:
        return parts[0], "MEK1", "MEK2", None

    # Handle CDK4_6 or CDK4_6-like tokens
    # Example: NRAS_CDK4_6  -> ["NRAS","CDK4","6"]  => CDK4/CDK6
    if len(parts) >= 3 and parts[1] == "CDK4" and parts[2] in {"6", "CDK6"}:
        return parts[0], "CDK4", "CDK6", None

    if len(parts) == 2:
        return parts[0], parts[1], None, None

    if len(parts) == 3:
        return parts[0], parts[1], parts[2], None

    # Fallback: keep first, try second, ignore rest
    return parts[0], (parts[1] if len(parts) > 1 else None), None, None


# =========================
# Input loading
# =========================
def load_screen_file(
    screen_name: str,
    sl_pair: str,
    aggregated: bool,
    base_dir: str = INPUT_SCREENS_DIR
) -> pd.DataFrame:
    """
    Keeps EXACT naming formats:
      - non-aggregated: {screen}_{sl_pair}_screen_with_hgnc.csv
      - aggregated:     {sl_pair}_{screen}_aggregated.csv

    Adds safe fallback:
      - if chosen path doesn't exist, try the other format automatically
    """
    screen_name = str(screen_name)
    sl_pair = str(sl_pair)

    nonagg_name = f"{screen_name}_{sl_pair}_screen_with_hgnc.csv"
    agg_name    = f"{sl_pair}_{screen_name}_aggregated.csv"

    nonagg_path = os.path.join(base_dir, nonagg_name)
    agg_path    = os.path.join(base_dir, agg_name)

    # 1) Primary path based on aggregated flag
    primary_path = agg_path if aggregated else nonagg_path
    secondary_path = nonagg_path if aggregated else agg_path

    if os.path.exists(primary_path):
        filepath = primary_path
        used_aggregated = aggregated
    elif os.path.exists(secondary_path):
        filepath = secondary_path
        used_aggregated = (not aggregated)
        print(f"[WARN] Config aggregated={aggregated} but file not found. "
              f"Using {'aggregated' if used_aggregated else 'non-aggregated'} file: {os.path.basename(filepath)}")
    else:
        raise FileNotFoundError(
            "Missing input file for BOTH naming formats:\n"
            f" - {nonagg_path}\n"
            f" - {agg_path}"
        )

    df = pd.read_csv(filepath)

    # Keep your behavior: Screen label nicer for non-aggregated
    df["Screen"] = (screen_name.capitalize() if not used_aggregated else screen_name)
    df["SL_Pair"] = sl_pair

    return df


def load_analysis_pairs(
    string_csv: str = ANALYSIS_PAIR_STRING,
    biogrid_csv: str = ANALYSIS_PAIR_BIOGRID
) -> pd.DataFrame:
    """
    Loads analysis_pair_string.csv and analysis_pair_biogrid.csv and produces a unique list of
    (screen, sl_pair, aggregated) rows.

    If only one file exists, it still works.
    If both exist, it merges conservatively (preferring explicit aggregated columns when present).
    """
    dfs = []
    for fp in [string_csv, biogrid_csv]:
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            meta = detect_columns(df)
            if meta["screen"] is None or meta["sl_pair"] is None:
                raise ValueError(
                    f"Could not detect required columns in {fp}. "
                    f"Need screen + sl_pair-like columns. Found: {list(df.columns)}"
                )

            out = df[[meta["screen"], meta["sl_pair"]]].copy()
            out.columns = ["screen", "sl_pair"]

            if meta["aggregated"] is not None:
                out["aggregated"] = df[meta["aggregated"]].map(parse_aggregated_value)
            else:
                # If the file doesn't have an aggregated flag, default False
                out["aggregated"] = False

            dfs.append(out)

    if not dfs:
        raise FileNotFoundError(
            f"Neither {string_csv} nor {biogrid_csv} found in the working directory."
        )

    merged = pd.concat(dfs, ignore_index=True)

    # If duplicates disagree on aggregated, prefer True (safer: loads aggregated filename when flagged anywhere)
    merged = (
        merged.groupby(["screen", "sl_pair"], as_index=False)["aggregated"]
        .max()
    )

    return merged




# =========================
# Transformation to ML table
# =========================
def standardize_query_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize query columns to:
      Query, hgnc_id_query, ensembl_gene_id_query, entrez_id_query
    while preserving your original intent.

    - Gene_corrected -> Query
    - hgnc_id -> hgnc_id_query
    - ensembl_gene_id -> ensembl_gene_id_query
    - entrez_id -> entrez_id_query
    - drops "Gene" if present
    """
    df = df.copy()

    ren = {}
    if "Gene_corrected" in df.columns:
        ren["Gene_corrected"] = "Query"
    elif "Query" not in df.columns and "Gene" in df.columns:
        # fallback
        ren["Gene"] = "Query"

    if "hgnc_id" in df.columns:
        ren["hgnc_id"] = "hgnc_id_query"
    if "ensembl_gene_id" in df.columns:
        ren["ensembl_gene_id"] = "ensembl_gene_id_query"
    if "entrez_id" in df.columns:
        ren["entrez_id"] = "entrez_id_query"

    df = df.rename(columns=ren)

    if "Gene" in df.columns and "Query" in df.columns:
        # avoid duplicate if Gene was not used as fallback
        if "Gene" != "Query":
            df = df.drop(columns=["Gene"], errors="ignore")

    return df


def add_biomarker_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Biomarker/Target1/Target2 from SL_Pair using your split rules.
    """
    df = df.copy()
    parsed = df["SL_Pair"].apply(split_sl_pair)
    df["Biomarker"] = parsed.apply(lambda x: x[0])
    df["Target1"] = parsed.apply(lambda x: x[1])
    df["Target2"] = parsed.apply(lambda x: x[2])
    df["Target3"] = parsed.apply(lambda x: x[3])
    return df


def add_id_columns_from_lookup(df: pd.DataFrame, hgnc_file: str = HGNC_FILE) -> pd.DataFrame:
    """
    Add HGNC/Ensembl/Entrez IDs for Biomarker/Target1/Target2/Target3 using the shared
    lib.hgnc_lookup resolver (strict 3-tier symbol > prev_symbol > alias, case-insensitive).
    """
    df = df.copy()

    role_to_col = {
        "biomarker": "Biomarker",
        "target1": "Target1",
        "target2": "Target2",
        "target3": "Target3",
    }

    # One shared lookup over every biomarker/target symbol present.
    genes = set()
    for gene_col in role_to_col.values():
        genes.update(df[gene_col].dropna().astype(str))
    lkp = build_hgnc_lookup(genes, hgnc_file)   # DataFrame indexed by gene

    def _ids(sym):
        if isinstance(sym, str) and sym in lkp.index:
            r = lkp.loc[sym]
            return (r["hgnc_id"], r["ensembl_gene_id"], r["entrez_id"])
        return (None, None, None)

    for role, gene_col in role_to_col.items():
        triples = df[gene_col].apply(_ids)
        df[f"hgnc_id_{role}"] = triples.apply(lambda t: t[0])
        df[f"ensembl_gene_id_{role}"] = triples.apply(lambda t: t[1])
        df[f"entrez_id_{role}"] = triples.apply(lambda t: t[2])

    return df


def add_feature_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure placeholder feature columns exist (set to None if missing).
    """
    df = df.copy()
    for c in FEATURE_PLACEHOLDER_COLS:
        if c not in df.columns:
            df[c] = None
    return df


def reorder_columns_like_notebook(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mimics your column_order cell but safely (only keeps columns that exist).
    """
    df = df.copy()

    desired_front = (
        ["Screen", "SL_Pair"] +
        ["Query", "Biomarker", "Target1", "Target2", "Target3",
         "entrez_id_query", "hgnc_id_query", "ensembl_gene_id_query",
         "entrez_id_biomarker", "hgnc_id_biomarker", "ensembl_gene_id_biomarker",
         "entrez_id_target1", "hgnc_id_target1", "ensembl_gene_id_target1",
         "entrez_id_target2", "hgnc_id_target2", "ensembl_gene_id_target2",
         "entrez_id_target3", "hgnc_id_target3", "ensembl_gene_id_target3"]
    )

    front = [c for c in desired_front if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


# =========================
# Orchestration
# =========================
def build_main_ml_dataset() -> pd.DataFrame:
    """
    Full pipeline:
      - load analysis pairs (screen/sl_pair/aggregated)
      - load each screen file using the exact naming formats
      - concat
      - standardize Query columns
      - add Biomarker/Target columns
      - add biomarker/target IDs from HGNC lookup
      - add placeholder ML feature columns
      - reorder columns
    """
    pairs_df = load_analysis_pairs(ANALYSIS_PAIR_STRING, ANALYSIS_PAIR_BIOGRID)

    dfs = []
    for _, row in pairs_df.iterrows():
        df = load_screen_file(
            screen_name=str(row["screen"]),
            sl_pair=str(row["sl_pair"]),
            aggregated=bool(row["aggregated"]),
            base_dir=INPUT_SCREENS_DIR
        )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    combined = standardize_query_columns(combined)
    combined = add_biomarker_target_columns(combined)
    combined = add_id_columns_from_lookup(combined, HGNC_FILE)
    combined = add_feature_placeholders(combined)
    combined = reorder_columns_like_notebook(combined)

    return combined


def write_main_dataset(df: pd.DataFrame, output_file: str = OUTPUT_MAIN_FILE) -> None:
    ensure_dir(os.path.dirname(output_file))
    df.to_csv(output_file, index=False)
    print(f"[OK] Wrote: {output_file}")
    print(f"[OK] Rows: {len(df):,} | Cols: {df.shape[1]:,}")


# =========================
# Run
# =========================
df_main = build_main_ml_dataset()
write_main_dataset(df_main, OUTPUT_MAIN_FILE)

df_main