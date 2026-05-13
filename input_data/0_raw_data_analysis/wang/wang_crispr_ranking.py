import pandas as pd
import numpy as np
import os
from typing import Optional


CONTROL_TERMS = [
    "NON-TARGET", "NON TARGET", "NTC", "CONTROL",
    "SCRAMBLE", "SAFE", "INTERGENIC", "LACZ", "EGFP",
]
MIR_PREFIXES = ("HSA-MIR", "MIR")

HGNC_FILE  = "../../HGNC/hgnc_complete_set.txt"
CRISPR_FILE = "wang_raw_data.xlsx"
OUTPUT_DIR  = "."


def build_hgnc_lookup(hgnc_path: str) -> dict:
    """
    Build a symbol → HGNC-ID lookup from the HGNC complete-set TSV.

    The lookup covers:
      - official 'symbol' column
      - all tokens in 'prev_symbol' (pipe-separated)
      - all tokens in 'alias_symbol' (pipe-separated)

    All keys are uppercased and stripped so matching is case-insensitive.

    Parameters
    ----------
    hgnc_path : str
        Path to the HGNC complete-set TSV file.

    Returns
    -------
    dict
        Mapping of UPPER_SYMBOL -> hgnc_id (string).
    """
    hgnc = pd.read_csv(hgnc_path, sep="\t", low_memory=False, dtype=str)
    lookup = {}

    for _, row in hgnc.iterrows():
        hgnc_id = str(row.get("hgnc_id", "")).strip()
        if not hgnc_id or hgnc_id == "nan":
            continue

        # Collect all symbol tokens for this gene
        tokens = []

        sym = str(row.get("symbol", "")).strip()
        if sym and sym != "nan":
            tokens.append(sym)

        for col in ("prev_symbol", "alias_symbol"):
            raw = str(row.get(col, "")).strip()
            if raw and raw != "nan":
                tokens.extend([t.strip() for t in raw.split("|")])

        for tok in tokens:
            key = tok.upper()
            if key:
                # First-seen wins (official symbol rows come first in the file)
                lookup.setdefault(key, hgnc_id)

    print(f"[HGNC] Loaded {len(lookup):,} symbol tokens from {hgnc_path}")
    return lookup



def clean_gene(symbol) -> Optional[str]:
    """
    Clean a raw sgRNA-Target gene symbol.

    Steps:
      1. Return None for actual NaN / float-NaN values.
      2. Strip whitespace and uppercase.
      3. Return None for empty strings.

    Parameters
    ----------
    symbol : any
        Raw value from the 'sgRNA Target' column.

    Returns
    -------
    str or None
        Cleaned uppercase symbol, or None if the value should be dropped.
    """
    if symbol is None:
        return None
    if isinstance(symbol, float) and np.isnan(symbol):
        return None
    cleaned = str(symbol).strip().upper()
    return cleaned if cleaned else None



def drop_controls(df: pd.DataFrame, gene_col: str) -> pd.DataFrame:
    """
    Remove sgRNA rows that target controls or non-protein-coding loci.

    Drops rows where the (already uppercased) gene symbol:
      - is NaN / None / empty
      - contains any CONTROL_TERMS substring (case-insensitive; genes are
        already uppercased so we compare against uppercase terms)
      - starts with any MIR_PREFIXES  (e.g. 'MIR', 'HSA-MIR')

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a cleaned gene-symbol column.
    gene_col : str
        Name of the column holding cleaned gene symbols.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame (copy with reset index).
    """
    mask_valid = df[gene_col].notna() & (df[gene_col] != "")

    # Build a control-substring mask
    control_upper = [t.upper() for t in CONTROL_TERMS]
    def _is_control(sym):
        if not sym:
            return True
        for term in control_upper:
            if term in sym:
                return True
        for prefix in MIR_PREFIXES:
            if sym.startswith(prefix):
                return True
        return False

    mask_not_control = ~df[gene_col].apply(_is_control)
    filtered = df[mask_valid & mask_not_control].copy()
    return filtered.reset_index(drop=True)



def process_sheet(
    sheet_df: pd.DataFrame,
    sheet_name: str,
    hgnc_lookup: dict,
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Full pipeline for one CRISPR screen sheet.

    Steps
    -----
    1. Validate required input columns.
    2. Clean gene symbols ('sgRNA Target' → 'Gene_clean').
    3. Drop control / non-gene rows via drop_controls().
    4. Convert 'Log2(Fold-change)' to numeric; drop NaN rows.
    5. Compute enrichment threshold using ALL post-control sgRNAs
       (mu + 4*sd, as described in the paper) BEFORE HGNC filtering.
    6. HGNC filter: keep only rows whose gene maps to an HGNC ID.
    7. Aggregate per gene: n_total, n_enriched, frac_enriched,
       majority_enriched, mean_enriched_log2fc, mean_log2fc_all.
    8. Sort genes: majority_enriched ↓, frac_enriched ↓, n_enriched ↓,
       mean_enriched_log2fc ↓ (NaN last), mean_log2fc_all ↓,
       n_total ↓, Gene ↑.
    9. Label top-{top_n} genes 'Resistance'; rest 'Non-Resistance'.
    10. Print per-sheet summary.

    Parameters
    ----------
    sheet_df : pd.DataFrame
        Raw DataFrame for one sheet of the CRISPR workbook.
    sheet_name : str
        Sheet name (used for display only).
    hgnc_lookup : dict
        UPPER_SYMBOL → hgnc_id mapping from build_hgnc_lookup().
    top_n : int
        Number of top-ranked genes to label 'Resistance' (default 100).

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame: ['Gene', 'Label'] with one row per gene,
        no duplicates.
    """
    df = sheet_df.copy()

    # --- Step 1: validate required columns ---
    required_cols = ["sgRNA Target", "Log2(Fold-change)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{sheet_name}] Missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # --- Step 2: clean gene symbols ---
    df["Gene_clean"] = df["sgRNA Target"].apply(clean_gene)

    # --- Step 3: drop controls ---
    df = drop_controls(df, "Gene_clean")

    # --- Step 4: numeric Log2FC, drop NA ---
    df["Log2FC"] = pd.to_numeric(df["Log2(Fold-change)"], errors="coerce")
    df = df.dropna(subset=["Log2FC"]).reset_index(drop=True)

    # --- Step 5: enrichment threshold (computed over ALL post-control sgRNAs,
    #             before HGNC filtering, as per the paper) ---
    mu     = df["Log2FC"].mean()
    sd     = df["Log2FC"].std(ddof=1)
    cutoff = mu + 4 * sd
    df["is_enriched"] = df["Log2FC"] >= cutoff
    print(
        f"[{sheet_name}] Enrichment threshold (pre-HGNC): mean={mu:.4f}, "
        f"sd={sd:.4f}, cutoff (mu+4sd)={cutoff:.4f}"
    )

    # --- Step 6: HGNC filter ---
    df["HGNC_ID"] = df["Gene_clean"].map(hgnc_lookup)
    n_before_hgnc = len(df)
    df = df.dropna(subset=["HGNC_ID"]).reset_index(drop=True)
    n_after_hgnc = len(df)
    print(
        f"[{sheet_name}] sgRNAs before HGNC filter: {n_before_hgnc:,} "
        f"→ after: {n_after_hgnc:,} "
        f"(dropped {n_before_hgnc - n_after_hgnc:,} with no HGNC ID)"
    )

    # --- Step 7: aggregate per gene ---
    agg = (
        df.groupby("Gene_clean", sort=False)
        .agg(
            n_total         = ("Log2FC", "count"),
            n_enriched      = ("is_enriched", "sum"),
            mean_log2fc_all = ("Log2FC", "mean"),
        )
        .reset_index()
        .rename(columns={"Gene_clean": "Gene"})
    )
    agg["n_enriched"] = agg["n_enriched"].astype(int)
    agg["frac_enriched"] = agg["n_enriched"] / agg["n_total"]
    agg["majority_enriched"] = agg["frac_enriched"] >= 0.5

    # mean_enriched_log2fc: mean of Log2FC among enriched sgRNAs only
    enriched_means = (
        df[df["is_enriched"]]
        .groupby("Gene_clean")["Log2FC"]
        .mean()
        .rename("mean_enriched_log2fc")
    )
    agg = agg.merge(enriched_means, left_on="Gene", right_index=True, how="left")

    # --- Step 8: sort ---
    # majority_enriched is bool (True sorts before False when ascending=False).
    # NaN in mean_enriched_log2fc goes last via na_position='last'.
    agg = agg.sort_values(
        by=[
            "majority_enriched",
            "frac_enriched",
            "n_enriched",
            "mean_enriched_log2fc",
            "mean_log2fc_all",
            "n_total",
            "Gene",
        ],
        ascending=[False, False, False, False, False, False, True],
        na_position="last",  # NaN in mean_enriched_log2fc goes to bottom
    ).reset_index(drop=True)

    # --- Step 8: label ---
    n_genes = len(agg)
    actual_top = min(top_n, n_genes)
    if actual_top < top_n:
        print(
            f"[{sheet_name}] NOTE: Only {n_genes} genes remain after filtering "
            f"(< {top_n}); all labeled 'Resistance'."
        )

    agg["Class"] = "Non-Resistance"
    agg.loc[agg.index[:actual_top], "Class"] = "Resistance"

    # --- Step 9: summary ---
    resistance_count     = (agg["Class"] == "Resistance").sum()
    non_resistance_count = (agg["Class"] == "Non-Resistance").sum()
    print(
        f"[{sheet_name}] Summary:\n"
        f"  Total sgRNAs after all filters : {n_after_hgnc:,}\n"
        f"  Total genes after HGNC filter  : {n_genes:,}\n"
        f"  Resistance genes               : {resistance_count}\n"
        f"  Non-Resistance genes           : {non_resistance_count}"
    )

    return agg[["Gene", "Class"]].copy()



os.makedirs(OUTPUT_DIR, exist_ok=True)

hgnc_lookup = build_hgnc_lookup(HGNC_FILE)

xl = pd.ExcelFile(CRISPR_FILE)
print(f"\nSheets found in {CRISPR_FILE}: {xl.sheet_names}\n")

screen_results = []  # collect per-screen DataFrames for Cell 8 merge

for sheet_name in xl.sheet_names:
    df_sheet = xl.parse(sheet_name)
    result   = process_sheet(df_sheet, sheet_name, hgnc_lookup)
    screen_results.append(result)

    out_path = os.path.join(OUTPUT_DIR, f"wang_{sheet_name}_KRAS_MEK_screen.xlsx")
    result.to_excel(out_path, index=False)
    print(f"[{sheet_name}] Saved → {out_path}\n")

print("Per-screen files done. Run Cell 8 to produce the merged union file.")



def merge_union(results: list, sheet_names: list) -> pd.DataFrame:
    """
    Merge per-screen [Gene, Class] DataFrames using a union rule.

    Union rule
    ----------
    - If a gene is labeled 'Resistance' in ANY screen → 'Resistance'.
    - If a gene is labeled 'Non-Resistance' in ALL screens it appears in
      → 'Non-Resistance'.
    - A gene absent from a screen (NaN after outer merge) is treated as
      neutral — its label is determined solely by the screens it did appear in.

    Genes appearing in only one screen receive that screen's label directly.
    No duplicate Gene rows are produced (outer merge on 'Gene' key).

    Parameters
    ----------
    results : list of pd.DataFrame
        Per-screen DataFrames, each with columns ['Gene', 'Class'].
    sheet_names : list of str
        Sheet names in the same order as `results` (used for print output).

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame: ['Class', 'Gene'], Class first, sorted so
        'Resistance' rows appear before 'Non-Resistance', one row per unique
        gene, no duplicates.
    """
    if not results:
        raise ValueError("results list is empty; nothing to merge.")

    # Start with the first screen; outer-merge each subsequent screen
    merged = results[0].rename(columns={"Class": f"Class_{sheet_names[0]}"})
    for i, df in enumerate(results[1:], start=1):
        right = df.rename(columns={"Class": f"Class_{sheet_names[i]}"})
        merged = pd.merge(merged, right, on="Gene", how="outer")

    # Apply union rule across all Class_* columns
    class_cols = [c for c in merged.columns if c.startswith("Class_")]
    merged["Class"] = merged[class_cols].apply(
        lambda row: "Resistance" if "Resistance" in row.values else "Non-Resistance",
        axis=1,
    )

    # Class first column; sort Resistance before Non-Resistance
    result = (
        merged[["Class", "Gene"]]
        .copy()
        .sort_values("Class", ascending=False)   # R > N alphabetically
        .reset_index(drop=True)
    )

    # Sanity check — no duplicates
    assert result["Gene"].is_unique, "Duplicate Gene rows detected after merge!"

    resistance_count     = (result["Class"] == "Resistance").sum()
    non_resistance_count = (result["Class"] == "Non-Resistance").sum()
    print(
        f"[Merged Union] Summary:\n"
        f"  Screens merged          : {sheet_names}\n"
        f"  Total unique genes      : {len(result):,}\n"
        f"  Resistance genes        : {resistance_count}\n"
        f"  Non-Resistance genes    : {non_resistance_count}"
    )

    return result


merged_result = merge_union(screen_results, xl.sheet_names)
merged_path   = os.path.join(OUTPUT_DIR, "wang_KRAS_MEK_screen.xlsx")
merged_result.to_excel(merged_path, index=False)
print(f"[Merged Union] Saved → {merged_path}")

print("\nAll done.")
