"""Knoll validation — feature SOURCING from clinical MTAP-PRMT5 (transition).

Instead of recomputing PPI/expression/essentiality features for the Knoll screen,
this pulls the already-computed feature columns from the clinical-trials
MTAP-PRMT5 dataset. The two feature sets were verified identical for matched
genes (same biomarker/target, same reference DBs), so sourcing from clinical
removes the redundant extraction while keeping Knoll's own gene set + labels.

  - Gene set + labels (Class):   Knoll  (features_main_schema.csv, from 01/02)
  - Feature values (18 cols):    clinical MTAP-PRMT5 withfeatures CSV

The original lib-based extraction is preserved in ../knoll_archive/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATA,
    FEATURES_MAIN_SCHEMA_CSV,
    FEATURES_DROPNA_DEDUP_CSV,
    FEATURE_COLS,
    TARGET_COL,
    SCORE_COL,
    CLINICAL_PRED_CSV,
)

# Source: clinical-trials MTAP-PRMT5 PREDICTION CSV (18 features + Resistance_Score).
# Both the features AND the model probability are sourced — no local retrain.
FEATURES_DROPNA_CSV = DATA / "features_dropna.csv"

MATCH_KEY = "hgnc_id_query"


def main() -> None:
    schema = pd.read_csv(FEATURES_MAIN_SCHEMA_CSV, low_memory=False)
    clin = pd.read_csv(CLINICAL_PRED_CSV, low_memory=False)
    print(f"Knoll schema rows: {len(schema)}  |  clinical MTAP-PRMT5 pred rows: {len(clin)}")

    # 1) Knoll metadata + labels (drop any placeholder feature columns from the schema)
    meta_cols = [c for c in schema.columns if c not in FEATURE_COLS]
    meta = schema[meta_cols].copy()

    # 2) The 18 feature columns + the clinical score, sourced from clinical, matched on HGNC id
    missing = [c for c in FEATURE_COLS + [SCORE_COL] if c not in clin.columns]
    if missing:
        raise ValueError(f"Clinical prediction CSV missing columns: {missing}")
    feat = (clin[[MATCH_KEY] + FEATURE_COLS + [SCORE_COL]]
            .dropna(subset=[MATCH_KEY]).drop_duplicates(MATCH_KEY))

    merged = meta.merge(feat, on=MATCH_KEY, how="left")

    # 3) Class_Processed from Knoll's Class labels (eval/panel relabel from labels_AvgDiff anyway)
    if "Class" in merged.columns:
        merged[TARGET_COL] = (merged["Class"] == "Resistance").astype(int)

    # 4) Drop Knoll-only genes that have no clinical features (e.g. pseudogenes /
    #    non-protein-coding genes absent from the clinical universe).
    n0 = len(merged)
    dropna = merged.dropna(subset=FEATURE_COLS)
    dropped = sorted(set(merged.loc[merged[FEATURE_COLS].isna().any(axis=1), "Query"].astype(str)))
    print(f"rows {n0} -> after dropna {len(dropna)} (dropped {n0 - len(dropna)} Knoll-only genes: {dropped})")
    dropna.to_csv(FEATURES_DROPNA_CSV, index=False)

    # 5) Dedup on gene
    dedup = dropna.drop_duplicates(subset=[MATCH_KEY]).reset_index(drop=True)
    dedup.to_csv(FEATURES_DROPNA_DEDUP_CSV, index=False)
    print(f"[OK] wrote {FEATURES_DROPNA_DEDUP_CSV}  rows={len(dedup)}  "
          f"(features + {SCORE_COL} sourced from clinical MTAP-PRMT5 predictions)")


if __name__ == "__main__":
    main()
