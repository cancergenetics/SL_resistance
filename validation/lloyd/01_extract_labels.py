"""Step 01 — Extract resistance labels from Lloyd 2021 MAGeCK output (KO IC90 SUM only).

Condition: ATM-KO cells, IC90 AZD6738, SUM analysis (paper's primary screen for
ATM-SL resistance — 8 FDR<0.1 hits vs 1 in REP).

Labeling: top-100 genes by pos|rank ascending → Resistance; rest → Non-Resistance.

Output: data/labels_KO_IC90_SUM.csv
        Columns: mouse_gene, screen_name, pos_rank, pos_lfc, pos_fdr, label
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA,
    EXCEL_PATH,
    KNOWN_HITS_MOUSE,
    LABELS_IC90_SUM_CSV,
    SCREEN_IC90_SUM,
    SHEET_IC90_SUM,
    TOP_N,
)


def load_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return df[["id", "pos|rank", "pos|lfc", "pos|p-value", "pos|fdr"]].copy()


def build_labels(df: pd.DataFrame, screen_name: str, resistance_genes: set[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "mouse_gene":  df["id"].values,
        "screen_name": screen_name,
        "pos_rank":    df["pos|rank"].values,
        "pos_lfc":     df["pos|lfc"].values,
        "pos_fdr":     df["pos|fdr"].values,
        "label":       ["Resistance" if g in resistance_genes else "Non-Resistance"
                        for g in df["id"]],
    })


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    print(f"Loading sheet '{SHEET_IC90_SUM}' from {EXCEL_PATH.name} …")
    ic90_sum = load_sheet(EXCEL_PATH, SHEET_IC90_SUM)

    res_ic90_sum = set(ic90_sum.nsmallest(TOP_N, "pos|rank")["id"])

    labels = build_labels(ic90_sum, SCREEN_IC90_SUM, res_ic90_sum)
    labels.to_csv(LABELS_IC90_SUM_CSV, index=False)

    counts = labels["label"].value_counts()
    print(f"\n{SCREEN_IC90_SUM}:")
    print(f"  Resistance={counts.get('Resistance', 0)}  "
          f"Non-Resistance={counts.get('Non-Resistance', 0)}  "
          f"Total={len(labels)}")

    print(f"\nKnown-hit check (top-{TOP_N} by pos|rank in IC90 SUM):")
    rank_ic90s = ic90_sum.set_index("id")["pos|rank"]
    for g in KNOWN_HITS_MOUSE:
        rank = int(rank_ic90s.get(g, float("nan")))
        mark = "✓" if g in res_ic90_sum else "✗"
        print(f"  {g:10s}  rank={rank:4d}  {mark}")

    print(f"\nwrote {LABELS_IC90_SUM_CSV}")


if __name__ == "__main__":
    main()
