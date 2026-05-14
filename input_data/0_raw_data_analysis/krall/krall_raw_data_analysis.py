import os
import re
import sys
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../lib")))
from hgnc_lookup import build_hgnc_lookup

INPUT_FILE = "krall_supp_file_1.xlsx"
OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../1_resistance_screens")
)
HGNC_TSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../HGNC/hgnc_complete_set.txt"))

SHEETS = {
    "BRAF_MEK STARS": "krall_BRAF_MEK_screen.xlsx",
    "KRAS_MEK STARS": "krall_KRAS_MEK_screen.xlsx",
    "NRAS_MEK STARS": "krall_NRAS_MEK_screen.xlsx",
}
TOP_N = 100


def serial_to_date_str(serial):
    return (datetime(1899, 12, 30) + timedelta(days=int(serial))).strftime("%d-%b")


def fix_date_gene(gene):
    """Convert Excel date-converted gene names back to HGNC-approved symbols."""
    # Integer serial → date string first
    if isinstance(gene, (int, float)) and not isinstance(gene, bool):
        gene = serial_to_date_str(int(gene))
    gene = str(gene).strip()
    m = re.match(r"^(\d+)-(Mar)$", gene)
    if m:
        return f"MARCHF{int(m.group(1))}"
    m = re.match(r"^(\d+)-(Sep)$", gene)
    if m:
        return f"SEPTIN{int(m.group(1))}"
    if gene in ("01-Dec", "30-Nov"):
        return "BHLHE40"
    return gene


def is_control(gene):
    g = str(gene).upper()
    if "NONTARGETINGCONTROLGUIDEFORHUMAN" in g.replace("-", "").replace("_", ""):
        return True
    if g.startswith("HSA-MIR") or g.startswith("HSA-LET"):
        return True
    if g.startswith("LOC") and g[3:].isdigit():
        return True
    return False


xl = pd.ExcelFile(INPUT_FILE)

for sheet, out_filename in SHEETS.items():
    df = xl.parse(sheet, header=1)
    print(f"\n=== {sheet} ===")
    print(f"Total genes (raw): {len(df)}")

    df = df[df["Gene Symbol"].notna()].copy()
    df["Gene Symbol"] = df["Gene Symbol"].apply(fix_date_gene)

    df = df[~df["Gene Symbol"].apply(is_control)].reset_index(drop=True)
    print(f"After filter: {len(df)}")

    # Deduplicate gene symbols — keep highest STARS Score (handles date-fixed duplicates e.g. BHLHE40)
    df = df.sort_values("STARS Score", ascending=False).drop_duplicates(subset="Gene Symbol").reset_index(drop=True)

    # HGNC validation
    lkp = build_hgnc_lookup(df["Gene Symbol"].tolist(), HGNC_TSV)
    no_hgnc = sorted(lkp[lkp["hgnc_id"].isna()].index.tolist())
    print(f"Genes with no HGNC mapping: {len(no_hgnc)}")
    if no_hgnc:
        print(f"  {no_hgnc}")

    df = df[~df["Gene Symbol"].isin(set(no_hgnc))].reset_index(drop=True)
    print(f"After HGNC filter: {len(df)}")

    df["Class"] = "Non-Resistance"
    df.loc[df.index[:TOP_N], "Class"] = "Resistance"

    res     = (df["Class"] == "Resistance").sum()
    non_res = (df["Class"] == "Non-Resistance").sum()
    print(f"Resistance: {res}, Non-Resistance: {non_res}, Total: {len(df)}")
    print(f"Top 5: {df['Gene Symbol'].iloc[:5].tolist()}")

    out = pd.concat([
        pd.DataFrame({"Class": "Resistance",     "Gene": sorted(df[df["Class"] == "Resistance"]["Gene Symbol"])}),
        pd.DataFrame({"Class": "Non-Resistance", "Gene": sorted(df[df["Class"] == "Non-Resistance"]["Gene Symbol"])}),
    ], ignore_index=True)

    out_path = os.path.join(OUTPUT_DIR, out_filename)
    out.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")
