"""Lloyd validation — RF eval, IC90_SUM (primary; excl ARID1A_ATR variant for leakage check)."""

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
    LABELS_IC90_SUM_HUMAN_CSV,
    RESULTS,
    RF_KW,
    SCREEN_IC90_SUM,
    TARGET_COL,
    TRAIN_CSV,
)
from lib.rf_eval import run_screen_eval


def main() -> None:
    run_screen_eval(
        train_csv=TRAIN_CSV,
        val_features_csv=FEATURES_DROPNA_DEDUP_CSV,
        labels_csv=LABELS_IC90_SUM_HUMAN_CSV,
        screen_name=SCREEN_IC90_SUM,
        screen_display_name="Lloyd",
        variant_label="IC90_SUM",
        out_plot=RESULTS / "lloyd_roc_pr",
        out_plot_excl=RESULTS / "lloyd_roc_pr",   # only excl variant runs (skip_full=True)
        out_json=RESULTS / "lloyd_eval.json",
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        rf_kw=RF_KW,
        excl_sl_pair=EXCL_SL_PAIR,
        known_hits=KNOWN_HITS_HUMAN,
        skip_full=True,
        rf_color="#2ca02c",   # green (Lloyd palette; distinct from Knoll's default blue)
    )


if __name__ == "__main__":
    main()