#!/usr/bin/env python
# coding: utf-8


import os
import glob
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# =========================
# Configuration
# =========================
CONFIG = {
    "training_csv": "../input_data/3_ML_outputs/datasets/PredictingSLResistanceFeatures_main_withfeatures_dropna_remove_duplicate.csv",
    "dataset_dir": "datasets",
    "predictions_dir": "predictions",
    "pairs_xlsx": "biomarker_target_genes.xlsx",
    "supplementary_xlsx": "predictions/supplementary_file_clinical_trial_predictions.xlsx",
    "score_column": "Resistance_Score",
    "target_column": "Class_Processed",
}

# Single-source feature definition — copied verbatim from 11_ML.ipynb.
# Any drift between training and clinical inference is caught by the
# assertion in `predict_for_pair`.
FEATURES_DICT: List[Dict[str, str]] = [
    {"name": "StringInteractionWithBiomarker", "label": "STRING score Q-B", "category": "PPI"},
    {"name": "StringInteractionWithTarget", "label": "STRING score Q-T", "category": "PPI"},
    {"name": "CoexpressionWithBiomarker", "label": "Coexpression Q-B", "category": "Expression"},
    {"name": "CoexpressionWithTarget", "label": "Coexpression Q-T", "category": "Expression"},
    {"name": "AvgExpression", "label": "Query Gene Expression (avg)", "category": "Expression"},
    {"name": "CoessentialityWithBiomarker", "label": "Coeesentialty Q-B", "category": "Essentiality"},
    {"name": "CoessentialityWithTarget", "label": "Coeesentialty Q-T", "category": "Essentiality"},
    {"name": "FET_SharedInteractors_Biomarker_BIOGRID", "label": "Shared PPI of Q-B (BIOGRID)", "category": "PPI"},
    {"name": "FET_SharedInteractors_Target_BIOGRID", "label": "Shared PPI of Q-T (BIOGRID)", "category": "PPI"},
    {"name": "FET_SharedInteractors_Biomarker_STRING", "label": "Shared PPI of Q-B (STRING)", "category": "PPI"},
    {"name": "FET_SharedInteractors_Target_STRING", "label": "Shared PPI of Q-T (STRING)", "category": "PPI"},
    {"name": "BIOGRIDPhysicalInteractionQueryBiomarker", "label": "BIOGRID Physical Interaction Q-B", "category": "PPI"},
    {"name": "BIOGRIDPhysicalInteractionQueryTarget", "label": "BIOGRID Physical Q-T", "category": "PPI"},
    {"name": "ExpressionVariance", "label": "Query Gene Expression (var)", "category": "Expression"},
    {"name": "EssentialityVariance", "label": "Query Gene Essentiality (var)", "category": "Essentiality"},
    {"name": "EssentialityAverage", "label": "Query Gene Essentiality (avg)", "category": "Essentiality"},
    {"name": "Essentiality_Percentage", "label": "Query Gene Essentiality (% of cell lines)", "category": "Essentiality"},
]

FEATURE_COLUMNS: List[str] = [f["name"] for f in FEATURES_DICT]



# =========================
# Utilities
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_rf_model() -> RandomForestClassifier:
    """Instantiate the RandomForestClassifier with the exact hyperparameters
    used in 11_ML.ipynb so clinical predictions stay aligned with the
    published training run."""
    return RandomForestClassifier(
        n_estimators=1000,
        max_features=1,
        max_depth=15,
        min_samples_leaf=4,
        random_state=42,
        class_weight={0: 1, 1: 5},
        n_jobs=-1,
    )


def train_rf_from_main_pipeline(
    training_csv: str,
    feature_columns: List[str],
    target_column: str,
) -> RandomForestClassifier:
    train_df = pd.read_csv(training_csv)
    missing = [c for c in feature_columns if c not in train_df.columns]
    if missing:
        raise ValueError(f"Training data missing feature columns: {missing}")
    if target_column not in train_df.columns:
        raise ValueError(f"Training data missing target column: {target_column}")

    X_train = train_df[feature_columns].values
    y_train = train_df[target_column].values

    rf = build_rf_model()
    rf.fit(X_train, y_train)
    print(
        f"[train] fit RF on {len(train_df):,} rows, "
        f"{len(feature_columns)} features, "
        f"positive fraction={y_train.mean():.3f}"
    )
    return rf



# =========================
# Prediction
# =========================
def predict_for_pair(
    model: RandomForestClassifier,
    sl_pair: str,
    feature_columns: List[str],
    dataset_dir: str,
    predictions_dir: str,
    score_column: str,
) -> pd.DataFrame:
    input_path = os.path.join(
        dataset_dir, f"PredictingSLResistanceFeatures_{sl_pair}_main_withfeatures.csv"
    )
    df = pd.read_csv(input_path, low_memory=False)

    missing = [c for c in feature_columns if c not in df.columns]
    assert not missing, f"{sl_pair}: missing feature columns for inference: {missing}"

    X = df[feature_columns].values
    df[score_column] = model.predict_proba(X)[:, 1]

    ensure_dir(predictions_dir)
    output_path = os.path.join(
        predictions_dir,
        f"PredictingSLResistanceFeatures_{sl_pair}_main_withfeatures_pred.csv",
    )
    df.to_csv(output_path, index=False)
    print(f"[OK] {output_path}  rows={len(df):,}  mean={df[score_column].mean():.3f}")
    return df


def predict_all_pairs(
    model: RandomForestClassifier,
    pairs_df: pd.DataFrame,
    feature_columns: List[str],
    dataset_dir: str,
    predictions_dir: str,
    score_column: str,
) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    for _, row in pairs_df.iterrows():
        biomarker = str(row["Biomarker"]).strip()
        target1 = str(row["Target"]).strip()
        sl_pair = f"{biomarker}_{target1}"
        results[sl_pair] = predict_for_pair(
            model=model,
            sl_pair=sl_pair,
            feature_columns=feature_columns,
            dataset_dir=dataset_dir,
            predictions_dir=predictions_dir,
            score_column=score_column,
        )
    return results



# =========================
# Supplementary XLSX assembly
# =========================
def assemble_supplementary_xlsx(
    results: Dict[str, pd.DataFrame],
    supplementary_xlsx: str,
) -> None:
    ensure_dir(os.path.dirname(supplementary_xlsx) or ".")
    with pd.ExcelWriter(supplementary_xlsx, engine="openpyxl") as writer:
        for sl_pair, df in results.items():
            sheet = sl_pair[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"[OK] wrote {supplementary_xlsx}  sheets={len(results)}")



# =========================
# Run
# =========================
rf_model = train_rf_from_main_pipeline(
    training_csv=CONFIG["training_csv"],
    feature_columns=FEATURE_COLUMNS,
    target_column=CONFIG["target_column"],
)

pairs_df = pd.read_excel(CONFIG["pairs_xlsx"])
print(f"[info] Loaded {len(pairs_df)} biomarker-target pairs")

results = predict_all_pairs(
    model=rf_model,
    pairs_df=pairs_df,
    feature_columns=FEATURE_COLUMNS,
    dataset_dir=CONFIG["dataset_dir"],
    predictions_dir=CONFIG["predictions_dir"],
    score_column=CONFIG["score_column"],
)

assemble_supplementary_xlsx(results, CONFIG["supplementary_xlsx"])

sample = next(iter(results))
results[sample][["Query", "Biomarker", "Target1", CONFIG["score_column"]]].head()