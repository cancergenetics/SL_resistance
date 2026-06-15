"""Paths, constants, and screen metadata for the Knoll 2025 PRMT5i (MTAP-PRMT5) validation."""

from __future__ import annotations

from pathlib import Path

ROOT           = Path(__file__).resolve().parents[2]
KNOLL          = ROOT / "validation" / "knoll"
RAW_DATA       = KNOLL / "raw_data"
DATA           = KNOLL / "data"
RESULTS        = KNOLL / "results"

BIOMARKER_GENE = "MTAP"
TARGET1_GENE   = "PRMT5"
TARGET2_GENE   = None
DRUG           = "MRTX1719/MRTX9768"
SL_PAIR        = "MTAP_PRMT5"
TOP_N          = 100

SCREEN_AVG = "Knoll_MTAP_PRMT5_AvgDiff"

RAW_CSV          = RAW_DATA / "knoll_raw_data.csv"
LABELS_AVG_CSV   = DATA / "labels_AvgDiff.csv"
LABELS_HUMAN_CSV = LABELS_AVG_CSV

CONTROL_GENES    = ["AAVS1", "nonTarget", "chr2"]
KNOWN_HITS_HUMAN = ["CARM1"]

HGNC_TSV = ROOT / "input_data" / "HGNC" / "hgnc_complete_set.txt"

MAIN_SCHEMA_CSV = (
    ROOT / "input_data" / "3_ML_outputs" / "datasets"
    / "PredictingSLResistanceFeatures_main.csv"
)
TRAIN_CSV = (
    ROOT / "input_data" / "3_ML_outputs" / "datasets"
    / "PredictingSLResistanceFeatures_main_withfeatures_dropna_remove_duplicate.csv"
)

FEATURES_MAIN_SCHEMA_CSV    = DATA / "features_main_schema.csv"
FEATURES_DROPNA_DEDUP_CSV   = DATA / "features_dropna_dedup.csv"

# Clinical-trials prediction CSV (features + Resistance_Score). Both the features
# AND the model probability are sourced from here — no local retrain.
SCORE_COL         = "Resistance_Score"
CLINICAL_PRED_CSV = (
    ROOT / "clinical_trials" / "predictions"
    / "PredictingSLResistanceFeatures_MTAP_PRMT5_main_withfeatures_pred.csv"
)

LABEL_ONLY_FEATURE_COLS = [
    "StringInteractionWithBiomarker",
    "StringInteractionWithTarget",
    "CoexpressionWithBiomarker",
    "CoexpressionWithTarget",
    "AvgExpression",
    "CoessentialityWithBiomarker",
    "CoessentialityWithTarget",
    "FET_SharedInteractors_Biomarker_BIOGRID",
    "FET_SharedInteractors_Target_BIOGRID",
    "FET_SharedInteractors_Biomarker_STRING",
    "FET_SharedInteractors_Target_STRING",
]

FEATURE_COLS = [
    "StringInteractionWithBiomarker",
    "StringInteractionWithTarget",
    "CoexpressionWithBiomarker",
    "CoexpressionWithTarget",
    "AvgExpression",
    "CoessentialityWithBiomarker",
    "CoessentialityWithTarget",
    "FET_SharedInteractors_Biomarker_BIOGRID",
    "FET_SharedInteractors_Target_BIOGRID",
    "FET_SharedInteractors_Biomarker_STRING",
    "FET_SharedInteractors_Target_STRING",
    "ExpressionVariance",
    "EssentialityVariance",
    "EssentialityAverage",
    "BIOGRIDPhysicalInteractionQueryBiomarker",
    "BIOGRIDPhysicalInteractionQueryTarget",
    "BiomarkerType",
    "Essentiality_Percentage",
]

TARGET_COL = "Class_Processed"

RF_KW = dict(
    n_estimators=300,
    max_features="sqrt",
    max_depth=None,
    min_samples_leaf=4,
    random_state=42,
    class_weight={0: 1, 1: 10},
    n_jobs=-1,
)

EXCL_SL_PAIR = "MTAP_PRMT5"
