"""Lloyd validation — feature extraction (PPI + expression + essentiality + merge).

Thin wrapper that imports config + calls shared lib functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add submission root (lib/ lives there) and this folder (config.py) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    DATA,
    FEATURE_OUTPUT,
    FEATURES_MAIN_SCHEMA_CSV,
    HGNC_TSV,
    ROOT,
)
from lib.feature_extraction_ppi import extract_ppi
from lib.feature_extraction_expression import extract_expression
from lib.feature_extraction_essentiality import extract_essentiality
from lib.merge_features import merge_features


def main() -> None:
    print("=== PPI features ===")
    extract_ppi(FEATURES_MAIN_SCHEMA_CSV, FEATURE_OUTPUT, HGNC_TSV, ROOT)

    print("\n=== Expression features ===")
    extract_expression(FEATURES_MAIN_SCHEMA_CSV, FEATURE_OUTPUT, ROOT)

    print("\n=== Essentiality features ===")
    extract_essentiality(FEATURES_MAIN_SCHEMA_CSV, FEATURE_OUTPUT, ROOT)

    print("\n=== Merge ===")
    merge_features(FEATURES_MAIN_SCHEMA_CSV, FEATURE_OUTPUT, DATA)


if __name__ == "__main__":
    main()