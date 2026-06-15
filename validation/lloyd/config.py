"""Paths, constants, and screen metadata for the Lloyd 2021 ATRi resistance validation."""

from __future__ import annotations

from pathlib import Path

ROOT           = Path(__file__).resolve().parents[2]
LLOYD          = ROOT / "validation" / "lloyd"
RAW_DATA       = LLOYD / "raw_data"
DATA           = LLOYD / "data"
RESULTS        = LLOYD / "results"

BIOMARKER_GENE = "ATM"
TARGET1_GENE   = "ATR"
TARGET2_GENE   = None
DRUG           = "AZD6738"
SL_PAIR        = "ATM_ATR"
TOP_N          = 100

# IC90 variants (strong drug pressure — primary)
SCREEN_IC90_SUM   = "Lloyd_ATM_ATR_KO_IC90_SUM"
SCREEN_IC90_REP   = "Lloyd_ATM_ATR_KO_IC90_REP"
SCREEN_IC90_UNION = "Lloyd_ATM_ATR_KO_IC90_union"

# IC10 variants (low drug pressure — secondary)
SCREEN_IC10_SUM   = "Lloyd_ATM_ATR_KO_IC10_SUM"
SCREEN_IC10_REP   = "Lloyd_ATM_ATR_KO_IC10_REP"
SCREEN_IC10_UNION = "Lloyd_ATM_ATR_KO_IC10_union"

EXCEL_PATH     = RAW_DATA / "Supplementary Table 1.xlsx"
SHEET_IC90_SUM = "KO IC90 (SUM <30ex)"
SHEET_IC90_REP = "KO IC90 (REP <10ex)"
SHEET_IC10_SUM = "KO IC10 (SUM<30ex)"   # no space before <
SHEET_IC10_REP = "KO IC10 (REP<10ex)"   # no space before <

# Mouse-symbol label files (from step 01)
LABELS_IC90_SUM_CSV   = DATA / "labels_KO_IC90_SUM.csv"
LABELS_IC90_REP_CSV   = DATA / "labels_KO_IC90_REP.csv"
LABELS_IC90_UNION_CSV = DATA / "labels_KO_IC90_union.csv"
LABELS_IC10_SUM_CSV   = DATA / "labels_KO_IC10_SUM.csv"
LABELS_IC10_REP_CSV   = DATA / "labels_KO_IC10_REP.csv"
LABELS_IC10_UNION_CSV = DATA / "labels_KO_IC10_union.csv"

# Human-ortholog label files (from step 02)
LABELS_IC90_SUM_HUMAN_CSV   = DATA / "labels_KO_IC90_SUM_human.csv"
LABELS_IC90_REP_HUMAN_CSV   = DATA / "labels_KO_IC90_REP_human.csv"
LABELS_IC90_UNION_HUMAN_CSV = DATA / "labels_KO_IC90_union_human.csv"
LABELS_IC10_SUM_HUMAN_CSV   = DATA / "labels_KO_IC10_SUM_human.csv"
LABELS_IC10_REP_HUMAN_CSV   = DATA / "labels_KO_IC10_REP_human.csv"
LABELS_IC10_UNION_HUMAN_CSV = DATA / "labels_KO_IC10_union_human.csv"

# Primary human labels used for feature extraction (IC90 SUM — paper's primary screen)
LABELS_HUMAN_CSV = LABELS_IC90_SUM_HUMAN_CSV

MOUSE_HUMAN_MAP_CSV   = DATA / "mouse_to_human_map.csv"
ENSEMBL_ORTHOLOGS_TSV = ROOT / "input_data" / "Ensembl" / "mouse_human_orthologs_ensembl.tsv"
HGNC_TSV              = ROOT / "input_data" / "HGNC" / "hgnc_complete_set.txt"

KNOWN_HITS_MOUSE = ["Cdk8", "Ccnc", "E2f8"]
KNOWN_HITS_HUMAN = ["CDK8", "CCNC", "E2F8"]

# Reference ML schema and training data
MAIN_SCHEMA_CSV = (
    ROOT / "input_data" / "3_ML_outputs" / "datasets"
    / "PredictingSLResistanceFeatures_main.csv"
)
TRAIN_CSV = (
    ROOT / "input_data" / "3_ML_outputs" / "datasets"
    / "PredictingSLResistanceFeatures_main_withfeatures_dropna_remove_duplicate.csv"
)

# Feature pipeline outputs
FEATURES_MAIN_SCHEMA_CSV    = DATA / "features_main_schema.csv"
FEATURES_DROPNA_DEDUP_CSV   = DATA / "features_dropna_dedup.csv"

# Clinical-trials prediction CSV (features + Resistance_Score). Both the features
# AND the model probability are sourced from here — no local retrain. The clinical
# ATM_ATR model already excludes ARID1A_ATR (leakage), matching Lloyd's excl rule.
SCORE_COL         = "Resistance_Score"
CLINICAL_PRED_CSV = (
    ROOT / "clinical_trials" / "predictions"
    / "PredictingSLResistanceFeatures_ATM_ATR_main_withfeatures_pred.csv"
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

EXCL_SL_PAIR = "ARID1A_ATR"
