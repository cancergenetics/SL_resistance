import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../lib")))
from hgnc_lookup import build_hgnc_lookup

INPUT_FILE = "clements_supp_data_2.xlsx"
OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../1_resistance_screens/clements_BRCA2_PARP1_screen.xlsx")
)
HGNC_TSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../HGNC/hgnc_complete_set.txt"))
FDR_THR = 0.05

df = pd.read_excel(INPUT_FILE, sheet_name="Gene rank")
print(f"Loaded: {df.shape[0]} genes")

resistance = df[df["FDR positive selection"] < FDR_THR]["Gene name"].tolist()
non_resistance = df[df["FDR positive selection"] >= FDR_THR]["Gene name"].tolist()

print(f"Resistance (FDR < {FDR_THR}): {len(resistance)}")
print(f"Non-Resistance:               {len(non_resistance)}")
print(f"Total:                        {len(resistance) + len(non_resistance)}")

# HGNC validation
all_genes = resistance + non_resistance
lkp = build_hgnc_lookup(all_genes, HGNC_TSV)
no_hgnc = sorted(lkp[lkp["hgnc_id"].isna()].index.tolist())

print(f"\nGenes with no HGNC mapping: {len(no_hgnc)}")
if no_hgnc:
    print(f"  {no_hgnc}")

# Filter
no_hgnc_set = set(no_hgnc)
resistance    = [g for g in resistance    if g not in no_hgnc_set]
non_resistance = [g for g in non_resistance if g not in no_hgnc_set]

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
