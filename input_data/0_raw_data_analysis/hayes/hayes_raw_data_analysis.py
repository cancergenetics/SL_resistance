import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../lib")))
from hgnc_lookup import build_hgnc_lookup

INPUT_FILE = "hayes_supp_table_1.xlsx"
OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../1_resistance_screens")
)
HGNC_TSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../HGNC/hgnc_complete_set.txt"))

SHEETS = {
    "NRAS_MEK":    "hayes_NRAS_MEK_screen.xlsx",
    "NRAS_CDK4_6": "hayes_NRAS_CDK4_6_screen.xlsx",
}
TOP_N = 150


def is_control(gene):
    g = str(gene).upper()
    return (
        g.startswith("NO_CURRENT_")
        or g.startswith("LOC")
        or g.startswith("LINC")
    )


xl = pd.ExcelFile(INPUT_FILE)

for sheet, out_filename in SHEETS.items():
    df = xl.parse(sheet, converters={"GENE SYMBOL": str})
    print(f"\n=== {sheet} ===")
    print(f"Total sgRNAs: {len(df)}")

    # Remove controls
    df = df[~df["GENE SYMBOL"].apply(is_control)].copy()
    print(f"After control filter: {len(df)}")

    # Numeric columns
    df["Median LFC"] = pd.to_numeric(df["Median LFC"], errors="coerce")
    df["Z-score"]    = pd.to_numeric(df["Z-score"],    errors="coerce")
    df = df.dropna(subset=["Median LFC", "Z-score"]).reset_index(drop=True)

    # Aggregate per gene
    agg = df.groupby("GENE SYMBOL").agg(
        n_sgrna       = ("Z-score",    "count"),
        mean_zscore   = ("Z-score",    "mean"),
        median_lfc    = ("Median LFC", "median"),
        frac_enriched = ("Median LFC", lambda x: (x > 0).mean()),
    ).reset_index().rename(columns={"GENE SYMBOL": "Gene"})

    # Filter genes with < 2 sgRNAs
    before = len(agg)
    agg = agg[agg["n_sgrna"] >= 2].reset_index(drop=True)
    print(f"Genes after aggregation: {before} → after sgRNA≥2 filter: {len(agg)}")

    # HGNC validation
    lkp = build_hgnc_lookup(agg["Gene"].tolist(), HGNC_TSV)
    no_hgnc = sorted(lkp[lkp["hgnc_id"].isna()].index.tolist())
    print(f"Genes with no HGNC mapping: {len(no_hgnc)}")
    if no_hgnc:
        print(f"  {no_hgnc}")

    agg = agg[~agg["Gene"].isin(set(no_hgnc))].reset_index(drop=True)
    print(f"After HGNC filter: {len(agg)}")

    # Sort by mean Z-score descending
    agg = agg.sort_values("mean_zscore", ascending=False).reset_index(drop=True)

    # Label top-N
    agg["Class"] = "Non-Resistance"
    agg.loc[agg.index[:TOP_N], "Class"] = "Resistance"

    res     = (agg["Class"] == "Resistance").sum()
    non_res = (agg["Class"] == "Non-Resistance").sum()
    print(f"Resistance: {res}, Non-Resistance: {non_res}, Total: {len(agg)}")
    print(f"Top 5: {agg['Gene'].iloc[:5].tolist()}")

    out = pd.concat([
        pd.DataFrame({"Class": "Resistance",     "Gene": sorted(agg[agg["Class"] == "Resistance"]["Gene"])}),
        pd.DataFrame({"Class": "Non-Resistance", "Gene": sorted(agg[agg["Class"] == "Non-Resistance"]["Gene"])}),
    ], ignore_index=True)

    out_path = os.path.join(OUTPUT_DIR, out_filename)
    out.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")
