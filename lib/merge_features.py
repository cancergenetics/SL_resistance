"""Shared library — multi-target feature combine helper (single source of truth).

The feature-merge step differs between pipelines (the main pipeline `10` merges one
combined feature table with up to three targets; `clinical_trials/05` merges per-pair
files with a single target), so each keeps its own merge body. The one piece of logic
they share — collapsing per-target feature columns into a single value by picking the
largest-magnitude available value — lives here.

  select_highest_or_available(*vals) -> pick the value with the largest |x| among the
                                        non-NaN values; NaN if all NaN.
  merge_targets(*series)             -> row-wise select_highest_or_available across N
                                        per-target series (used for Target1/2/3).
  merge_dual_target(s1, s2)          -> 2-target convenience alias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_highest_or_available(*vals):
    """Pick the value with the largest absolute magnitude across targets; NaN if all NaN."""
    avail = [v for v in vals if not pd.isna(v)]
    if not avail:
        return np.nan
    return max(avail, key=abs)


def merge_targets(*series_list):
    """Combine N per-target series row-wise via select_highest_or_available."""
    return [select_highest_or_available(*vals) for vals in zip(*series_list)]


def merge_dual_target(series1, series2):
    """Backward-compatible 2-target alias."""
    return merge_targets(series1, series2)
