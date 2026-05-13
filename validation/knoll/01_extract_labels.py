"""Step 01 — Extract Knoll 2025 PRMT5i resistance labels from raw CRISPR screen.

Pipeline:
  1. Drop paralog (digenic) rows: GuideTargetSymbol containing "_"
  2. Drop control sgRNAs: AAVS1, nonTarget, chr2
  3. Sort by AvgDiff descending (resistance signal: positive = MRTXi LFC > DMSO LFC)
  4. Top-100 → Resistance; rest → Non-Resistance

Note: Order matters — controls dropped BEFORE labeling so positives reflect 100
real single-gene hits. (In raw data sorted desc, no controls are in top-100, but
this guards against future raw-data variants.)

Output: data/labels_AvgDiff.csv  with columns
        query_gene, AvgDiff, DiffLU99, DiffSW1573, label, screen_name
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CONTROL_GENES,
    DATA,
    KNOWN_HITS_HUMAN,
    LABELS_AVG_CSV,
    RAW_CSV,
    SCREEN_AVG,
    TOP_N,
)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV)
    print(f"raw rows: {len(df)}")

    df = df[~df["GuideTargetSymbol"].str.contains("_", na=False)].copy()
    print(f"after paralog drop: {len(df)}")

    df = df[~df["GuideTargetSymbol"].isin(CONTROL_GENES)].copy()
    print(f"after control drop: {len(df)}")

    df = df.sort_values("AvgDiff", ascending=False).reset_index(drop=True)
    df["label"] = "Non-Resistance"
    df.iloc[:TOP_N, df.columns.get_loc("label")] = "Resistance"

    out = (df.rename(columns={"GuideTargetSymbol": "query_gene"})
             [["query_gene", "AvgDiff", "DiffLU99", "DiffSW1573", "label"]]
             .copy())
    out["screen_name"] = SCREEN_AVG

    out.to_csv(LABELS_AVG_CSV, index=False)

    counts = out["label"].value_counts()
    print(f"  Resistance:     {counts.get('Resistance', 0)}")
    print(f"  Non-Resistance: {counts.get('Non-Resistance', 0)}")
    print(f"  AvgDiff cutoff @ rank {TOP_N}: {out.iloc[TOP_N-1]['AvgDiff']:.4f}")
    print(f"  AvgDiff at rank 1:           {out.iloc[0]['AvgDiff']:.4f} ({out.iloc[0]['query_gene']})")

    print("\n  Known-hit check:")
    for g in KNOWN_HITS_HUMAN:
        row = out[out["query_gene"] == g]
        if len(row):
            rank = row.index[0] + 1
            print(f"    {g:8s} rank={rank:>4}  AvgDiff={row.iloc[0]['AvgDiff']:.4f}  "
                  f"label={row.iloc[0]['label']}")
        else:
            print(f"    {g:8s} NOT FOUND")

    print(f"\n  wrote {LABELS_AVG_CSV}")


if __name__ == "__main__":
    main()