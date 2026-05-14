"""
Dunn et al. — PTEN-AKT / PTEN-PIK3CB resistance screens
Cell lines: EVSA-T (PR+ER-), HCC-70 (TNBC), ZR-75-1 (ER+PR-)
Library: Yusa hCRISPR KO v1

Resistance genes sourced from supplementary Excel file (dunn_supp_file_1.xlsx):
  - Sheet 'AZD8186(PI3KB)':     PIK3CB inhibitor screen, 3 cell lines
  - Sheet 'capivasertib(AKT)':  AKT inhibitor screen, 3 cell lines

Each sheet has 3 columns (one per cell line). Union rule: gene is Resistance if
enriched in ANY of the 3 cell lines.

Pool: Yusa hCRISPR KO v1 library genes with >= 1 sgRNA (18,009 genes).
Yusa library file (yusa_hcrispr_ko_grnas.xlsx) shared with the Noordermeer screen.
Download: Yusa lab hCRISPR KO v1 library (Human V1 sheet, Gene column).
"""

import datetime
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../lib")))
from hgnc_lookup import build_hgnc_lookup

SUPP_FILE  = "dunn_supp_file_1.xlsx"
YUSA_FILE  = "yusa_hcrispr_ko_grnas.xlsx"
HGNC_TSV   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../HGNC/hgnc_complete_set.txt"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../1_resistance_screens"))

SCREENS = {
    "PIK3CB": {"sheet": "AZD8186(PI3KB)",    "out": "dunn_PTEN_PIK3CB_screen.xlsx"},
    "AKT":    {"sheet": "capivasertib(AKT)", "out": "dunn_PTEN_AKT_screen.xlsx"},
}
SGRNA_MIN = 1


def fix_date_gene(gene):
    """Convert Excel date-converted Yusa library gene names (day = gene number)."""
    if isinstance(gene, (datetime.datetime, pd.Timestamp)):
        month, day = gene.month, gene.day
        if month == 3:  return f"MARCHF{day}"
        if month == 9:  return f"SEPTIN{day}"
        if month == 12 and day == 1: return "BHLHE40"
        return str(gene)
    return str(gene).strip()


# ── Yusa pool ───────────────────────────────────────────────────────────────
print("Loading Yusa library...")
yusa_df = pd.read_excel(YUSA_FILE, sheet_name="Human v1")
yusa_df["Gene"] = yusa_df["Gene"].apply(fix_date_gene)
sgrna_counts = yusa_df["Gene"].value_counts()
yusa_pool = set(sgrna_counts[sgrna_counts >= SGRNA_MIN].index)
print(f"Yusa pool (≥{SGRNA_MIN} sgRNA): {len(yusa_pool)}")

# ── Process each screen ─────────────────────────────────────────────────────
xl = pd.ExcelFile(SUPP_FILE)

for drug, cfg in SCREENS.items():
    print(f"\n=== {drug} screen ({cfg['sheet']}) ===")
    df = xl.parse(cfg["sheet"])
    print(f"Shape: {df.shape}, Columns: {df.columns.tolist()}")

    # Union resistance genes across all cell line columns
    resistance = set()
    for col in df.columns:
        genes = df[col].dropna().astype(str).str.strip()
        genes = genes[genes != ""]
        print(f"  {col}: {len(genes)} genes")
        resistance |= set(genes)
    print(f"Union resistance: {len(resistance)}")

    # Pool = Yusa pool ∪ resistance (includes hits outside library)
    pool = yusa_pool | resistance
    non_resistance = pool - resistance
    print(f"Pool: {len(pool)}, Non-Resistance: {len(non_resistance)}")

    # HGNC validation
    lkp = build_hgnc_lookup(sorted(pool), HGNC_TSV)
    no_hgnc = sorted(lkp[lkp["hgnc_id"].isna()].index.tolist())
    print(f"Genes with no HGNC mapping: {len(no_hgnc)}")
    if no_hgnc:
        print(f"  {no_hgnc}")

    no_hgnc_set    = set(no_hgnc)
    resistance     = resistance     - no_hgnc_set
    non_resistance = non_resistance - no_hgnc_set

    print(f"After HGNC filter: Resistance={len(resistance)}, Non-Resistance={len(non_resistance)}, Total={len(resistance)+len(non_resistance)}")

    out = pd.concat([
        pd.DataFrame({"Class": "Resistance",     "Gene": sorted(resistance)}),
        pd.DataFrame({"Class": "Non-Resistance", "Gene": sorted(non_resistance)}),
    ], ignore_index=True)

    out_path = os.path.join(OUTPUT_DIR, cfg["out"])
    out.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")
