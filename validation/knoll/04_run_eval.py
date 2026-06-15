"""Knoll validation — RF eval (full + auto-excl MTAP_PRMT5 if present in training)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    EXCL_SL_PAIR,
    FEATURE_COLS,
    FEATURES_DROPNA_DEDUP_CSV,
    KNOWN_HITS_HUMAN,
    LABELS_AVG_CSV,
    RESULTS,
    RF_KW,
    SCORE_COL,
    SCREEN_AVG,
    TARGET_COL,
    TRAIN_CSV,
)
from lib.rf_eval import run_screen_eval


def main() -> None:
    # Scores sourced from clinical MTAP_PRMT5 predictions — no local retrain.
    run_screen_eval(
        train_csv=TRAIN_CSV,
        val_features_csv=FEATURES_DROPNA_DEDUP_CSV,
        labels_csv=LABELS_AVG_CSV,
        screen_name=SCREEN_AVG,
        screen_display_name="Knoll",
        variant_label="AvgDiff",
        out_plot=RESULTS / "knoll_roc_pr",
        out_json=RESULTS / "knoll_eval.json",
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        rf_kw=RF_KW,
        known_hits=KNOWN_HITS_HUMAN,
        score_col=SCORE_COL,
    )


if __name__ == "__main__":
    main()