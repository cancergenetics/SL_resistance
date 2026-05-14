import datetime
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../lib")))
from hgnc_lookup import build_hgnc_lookup

SUPP_FILE  = "noordermeer_supp_table_4.xlsx"
# Yusa hCRISPR KO v1 library — not a Noordermeer supplementary file.
# Download from: https://www.addgene.org/pooled-library/yusa-mouse-ko-v1/
# or the human version from the Yusa lab. File used: yusa_hcrispr_ko_grnas.xlsx
# Sheet: "Human v1", column "Gene" (one row per sgRNA).
YUSA_FILE  = "yusa_hcrispr_ko_grnas.xlsx"
OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../1_resistance_screens/noordermeer_BRCA1_PARP1_screen.xlsx")
)
HGNC_TSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../HGNC/hgnc_complete_set.txt"))

TKOv1_SHEETS = [
    "SUM149PT olaparib TKOv1",
    "RPE1 BRCA1-KO olapariob TKOv1",   # note: typo "olapariob" in original
]
TOP_N      = 20
SGRNA_MIN  = 2
SGRNA_MAX  = 6


def fix_date_gene(gene):
    """Convert Excel date-converted gene names (datetime objects) to HGNC symbols.

    TKOv1 encoding: day = gene family number (e.g. 2017-03-09 → MARCHF9).
    """
    if isinstance(gene, (datetime.datetime, pd.Timestamp)):
        month, day = gene.month, gene.day
        if month == 3:
            return f"MARCHF{day}"
        if month == 9:
            return f"SEPTIN{day}"
        if month == 12 and day == 1:
            return "BHLHE40"
    return str(gene).strip()


# ── TKOv1 screens ──────────────────────────────────────────────────────────
tkov1_pools = []
tkov1_top20 = []

xl = pd.ExcelFile(SUPP_FILE)

for sheet in TKOv1_SHEETS:
    df = xl.parse(sheet)
    print(f"\n=== {sheet} ===")
    print(f"Raw genes: {len(df)}")

    df["Gene"] = df["Gene"].apply(fix_date_gene)

    df["sgRNA"] = pd.to_numeric(df["sgRNA"], errors="coerce")
    df["beta_1|beta"] = pd.to_numeric(df["beta_1|beta"], errors="coerce")

    df = df[(df["sgRNA"] >= SGRNA_MIN) & (df["sgRNA"] <= SGRNA_MAX)].reset_index(drop=True)
    print(f"After sgRNA filter (2–6): {len(df)}")

    pool = set(df["Gene"])
    tkov1_pools.append(pool)

    top20 = set(df.sort_values("beta_1|beta", ascending=False).head(TOP_N)["Gene"])
    tkov1_top20.append(top20)
    print(f"Pool: {len(pool)}, Top-20: {sorted(top20)}")


# ── Yusa screen (SUM149PT talazoparib) ─────────────────────────────────────
print(f"\n=== SUM149PT talazoparib (Yusa) ===")

# Top-20 from Venn diagram sheet col index 6, rows 4–23 (1-indexed Excel rows)
venn = xl.parse("Venn diagram", header=None)
yusa_top20 = set(venn.iloc[5:25, 6].dropna().astype(str).str.strip().tolist())
print(f"Yusa top-20: {sorted(yusa_top20)}")

# Pool from Yusa library — genes with ≥2 sgRNAs
yusa_df = pd.read_excel(YUSA_FILE, sheet_name="Human v1")
sgrna_per_gene = yusa_df["Gene"].value_counts()
yusa_pool = set(sgrna_per_gene[sgrna_per_gene >= 2].index)
print(f"Yusa pool (≥2 sgRNAs): {len(yusa_pool)}")


# ── Union merge ─────────────────────────────────────────────────────────────
total_pool = tkov1_pools[0] | tkov1_pools[1] | yusa_pool
resistance  = tkov1_top20[0] | tkov1_top20[1] | yusa_top20
non_resistance = total_pool - resistance

print(f"\n=== Union ===")
print(f"Total pool: {len(total_pool)}")
print(f"Resistance (union top-20): {len(resistance)}")
print(f"Non-Resistance: {len(non_resistance)}")

# ── HGNC validation ─────────────────────────────────────────────────────────
lkp = build_hgnc_lookup(sorted(total_pool), HGNC_TSV)
no_hgnc = sorted(lkp[lkp["hgnc_id"].isna()].index.tolist())
print(f"\nGenes with no HGNC mapping: {len(no_hgnc)}")
if no_hgnc:
    print(f"  {no_hgnc}")

no_hgnc_set = set(no_hgnc)
resistance     = resistance     - no_hgnc_set
non_resistance = non_resistance - no_hgnc_set

print(f"\nAfter HGNC filter:")
print(f"Resistance:     {len(resistance)}")
print(f"Non-Resistance: {len(non_resistance)}")
print(f"Total:          {len(resistance) + len(non_resistance)}")

out = pd.concat([
    pd.DataFrame({"Class": "Resistance",     "Gene": sorted(resistance)}),
    pd.DataFrame({"Class": "Non-Resistance", "Gene": sorted(non_resistance)}),
], ignore_index=True)

out.to_excel(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
