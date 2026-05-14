"""
Gallo et al. 2022, Nature — CCNE1/PKMYT1, RP-6306 resistance screen
Cell line: FT282-hTERT (CCNE1-high), TKOv3 library, two clones (C3, C4)

Raw data: gallo_supp_table_1.xlsx (Supplementary Data 3)

DrugZ analysis
--------------
DrugZ (Colic et al., 2019) was run separately on the raw sgRNA count data
(one run per clone) to produce the TSV outputs read below.

To reproduce the DrugZ outputs from raw counts:
    python drugz.py \\
        -i <raw_counts.txt> \\
        -o drugz_C3.tsv \\
        -c <control_sample> \\
        -x <treatment_sample>

Run once for clone C3 → drugz_C3.tsv
Run once for clone C4 → drugz_C4.tsv

DrugZ GitHub: https://github.com/hart-lab/drugz

Resistance labeling
-------------------
Threshold: sumZ > 9 per clone (DrugZ's internally computed combined score).
Union rule: gene is Resistance if sumZ > 9 in C3 OR C4.

Date-conversion fix: DrugZ input Excel auto-converted MARCH/SEPT/DEC gene
symbols to dates. Reversed here:
  XX-Mar  → MARCHFXX  (HGNC approved: MARCHF1–MARCHF11)
  XX-Sep  → SEPTINXX  (HGNC approved: SEPTIN1–SEPTIN15)
  01-Dec  → BHLHE40   (DEC1 alias → BHLHE40, protein-coding)
"""

import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../lib")))
from hgnc_lookup import build_hgnc_lookup

C3_TSV   = "drugz_C3.tsv"
C4_TSV   = "drugz_C4.tsv"
OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../1_resistance_screens/gallo_CCNE1_PKMYT1_screen.xlsx")
)
HGNC_TSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../HGNC/hgnc_complete_set.txt"))
SUMZ_THR = 9


def fix_date_gene(name):
    m = re.match(r"^(\d+)-(Mar)$", name)
    if m:
        return f"MARCHF{int(m.group(1))}"
    m = re.match(r"^(\d+)-(Sep)$", name)
    if m:
        return f"SEPTIN{int(m.group(1))}"
    if name == "01-Dec":
        return "BHLHE40"
    return name


c3 = pd.read_csv(C3_TSV, sep="\t")
c4 = pd.read_csv(C4_TSV, sep="\t")

c3["GENE"] = c3["GENE"].apply(fix_date_gene)
c4["GENE"] = c4["GENE"].apply(fix_date_gene)

res_c3 = set(c3[c3["sumZ"] > SUMZ_THR]["GENE"])
res_c4 = set(c4[c4["sumZ"] > SUMZ_THR]["GENE"])
resistance = res_c3 | res_c4
pool = set(c3["GENE"]) | set(c4["GENE"])
non_res = pool - resistance

print(f"C3 sumZ>{SUMZ_THR}: {len(res_c3)}")
print(f"C4 sumZ>{SUMZ_THR}: {len(res_c4)}")
print(f"C3∩C4 overlap: {len(res_c3 & res_c4)} — {sorted(res_c3 & res_c4)}")
print(f"Pool: {len(pool)}, Resistance: {len(resistance)}, Non-Resistance: {len(non_res)}")

# HGNC validation
lkp = build_hgnc_lookup(sorted(pool), HGNC_TSV)
no_hgnc = sorted(lkp[lkp["hgnc_id"].isna()].index.tolist())
print(f"\nGenes with no HGNC mapping: {len(no_hgnc)}")
if no_hgnc:
    print(f"  {no_hgnc}")

no_hgnc_set = set(no_hgnc)
resistance = resistance - no_hgnc_set
non_res    = non_res    - no_hgnc_set

print(f"\nAfter HGNC filter:")
print(f"Resistance:     {len(resistance)}")
print(f"Non-Resistance: {len(non_res)}")
print(f"Total:          {len(resistance) + len(non_res)}")

out = pd.concat([
    pd.DataFrame({"Class": "Resistance",     "Gene": sorted(resistance)}),
    pd.DataFrame({"Class": "Non-Resistance", "Gene": sorted(non_res)}),
], ignore_index=True)

out.to_excel(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
