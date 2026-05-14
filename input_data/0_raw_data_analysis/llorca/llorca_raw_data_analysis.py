import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../lib")))
from hgnc_lookup import build_hgnc_lookup

INPUT_FILE = "llorca_supp_table_3.xlsx"
OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../1_resistance_screens/llorca_ARID1A_ATR_screen.xlsx")
)
HGNC_TSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../HGNC/hgnc_complete_set.txt"))
TOP_N = 100


def fix_date_gene(gene):
    """Convert Excel date-converted gene names back to HGNC-approved symbols.

    Encoding: year - 2000 = gene family number (e.g. 2002-03-01 → MARCHF2).
    Handles both pd.Timestamp objects and string representations.
    """
    if isinstance(gene, pd.Timestamp):
        year, month = gene.year, gene.month
    else:
        s = str(gene).strip()
        m = re.match(r"^(\d{4})-(\d{2})-\d{2}", s)
        if m:
            year, month = int(m.group(1)), int(m.group(2))
        else:
            return s
    num = year - 2000
    if month == 3:
        return f"MARCHF{num}"
    if month == 9:
        return f"SEPTIN{num}"
    if month == 12:
        return "BHLHE40"
    return str(gene).strip()


def is_loc(gene):
    return bool(re.match(r"^LOC\d+$", gene))


df = pd.read_excel(INPUT_FILE, sheet_name="Results_YCC6", header=1)
print(f"Raw genes: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Identify gene column (first column)
gene_col = df.columns[0]
print(f"Gene column: '{gene_col}'")

# Fix date-converted genes
df[gene_col] = df[gene_col].apply(fix_date_gene)

# Remove LOC* non-coding genes
before = len(df)
df = df[~df[gene_col].apply(is_loc)].reset_index(drop=True)
print(f"After LOC filter: {before} → {len(df)} (removed {before - len(df)})")

# Remove genes with num.sgRNAs <= 2
df["num.sgRNAs"] = pd.to_numeric(df["num.sgRNAs"], errors="coerce")
before = len(df)
df = df[df["num.sgRNAs"] > 2].reset_index(drop=True)
print(f"After sgRNA>2 filter: {before} → {len(df)} (removed {before - len(df)})")

# Sort by Rank.Prod ascending and deduplicate (keep best Rank.Prod per gene)
df["Rank.Prod"] = pd.to_numeric(df["Rank.Prod"], errors="coerce")
df = df.sort_values("Rank.Prod", ascending=True)
before = len(df)
df = df.drop_duplicates(subset=gene_col, keep="first").reset_index(drop=True)
print(f"After dedup: {before} → {len(df)} (removed {before - len(df)})")

# HGNC validation
lkp = build_hgnc_lookup(df[gene_col].tolist(), HGNC_TSV)
no_hgnc = sorted(lkp[lkp["hgnc_id"].isna()].index.tolist())
print(f"\nGenes with no HGNC mapping: {len(no_hgnc)}")
if no_hgnc:
    print(f"  {no_hgnc}")

df = df[~df[gene_col].isin(set(no_hgnc))].reset_index(drop=True)
print(f"After HGNC filter: {len(df)}")

# Label top-N
df["Class"] = "Non-Resistance"
df.loc[df.index[:TOP_N], "Class"] = "Resistance"

res     = (df["Class"] == "Resistance").sum()
non_res = (df["Class"] == "Non-Resistance").sum()
print(f"\nResistance: {res}, Non-Resistance: {non_res}, Total: {len(df)}")
print(f"Top 5: {df[gene_col].iloc[:5].tolist()}")

out = pd.concat([
    pd.DataFrame({"Class": "Resistance",     "Gene": sorted(df[df["Class"] == "Resistance"][gene_col])}),
    pd.DataFrame({"Class": "Non-Resistance", "Gene": sorted(df[df["Class"] == "Non-Resistance"][gene_col])}),
], ignore_index=True)

out.to_excel(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
