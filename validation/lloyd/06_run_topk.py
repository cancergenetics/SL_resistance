"""Lloyd validation — top-k prioritization for IC90_SUM (excl ARID1A_ATR from training)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    FEATURE_COLS,
    FEATURES_DROPNA_DEDUP_CSV,
    LABELS_IC90_SUM_HUMAN_CSV,
    RESULTS,
    RF_KW,
    SCORE_COL,
    SCREEN_IC90_SUM,
    TARGET_COL,
    TRAIN_CSV,
)
from lib.topk import run_screen_topk


def main() -> None:
    # Scores sourced from clinical ATM_ATR predictions (already excl ARID1A_ATR) — no retrain.
    run_screen_topk(
        train_csv=TRAIN_CSV,
        val_features_csv=FEATURES_DROPNA_DEDUP_CSV,
        labels_csv=LABELS_IC90_SUM_HUMAN_CSV,
        screen_name=SCREEN_IC90_SUM,
        out_stem=RESULTS / "topk_lloyd",
        out_json=RESULTS / "topk_lloyd.json",
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        rf_kw=RF_KW,
        score_col=SCORE_COL,
    )


if __name__ == "__main__":
    main()