# Predicting Resistance to Synthetic Lethal (SL) Therapies in Cancer

This repository contains Python scripts and files for data processing and analysis code for the paper of the same name, published in : Genome Medicine

Cite as : []

## Contents

- [Environment](#environment)
- [Data processing scripts overview](#data-processing-scripts-overview)
- [Validation screen analysis](#validation-screen-analysis)
- [Clinical trial dataset analysis](#clinical-trial-dataset-analysis)
- [Execution Order](#execution-order)
- [Data Sources](#data-sources)
- [Pre-generated Data Files](#pre-generated-data-files)

## Environment

The code was developed and tested using a Conda environment with **Python 3.12**.

To recreate the environment, run:

```bash
conda env create -f environment.yml
conda activate predicting_resistance_sl
```

## Data processing scripts overview

See [Execution Order](#execution-order) for how to run these scripts.

| Script | Figures | Brief description |
|--------|---------|-------------------|
| `01_resistance_screen_data_preprocessing.py`  | Fig. S1 | Preprocessing, mapping, and annotation of gene symbols in CRISPR resistance screens using HGNC, Entrez, and Ensembl identifiers. |
| `02_STRING_interaction_overlap_analysis.py`  | NA | Overlapping analysis of protein-protein interaction partners of SL pairs and resistance/non-resistance genes in STRING (medium confidence) database. |
| `03_BIOGRID_interaction_overlap_analysis.py` | NA | Overlapping analysis of protein-protein interaction partners of SL pairs and resistance/non-resistance genes in BIOGRID physical database. |
| `04_interaction_overlap_graphs.py`  | Fig. 1B, Fig. 2, Fig. S2-S3 | Visualization of overlapping analysis of protein-protein interaction partner enrichment of SL pairs among resistance/non-resistance genes across CRISPR screens. |
| `05_randomness_analysis.py`  | Fig. S4 | Random network analysis of resistance gene connectivity to synthetic lethal partners using degree-matched STRING and BioGRID PPI networks. |
| `06_ML_data_preprocessing.py`  | NA | Data preprocessing to generate a machine learning–ready dataset by combining CRISPR screen datasets for predictive modeling. |
| `07_feature_extraction_ppi_based.py`  | NA | PPI-based feature extraction for ML model. |
| `08_feature_extraction_expression_based.py` | NA | Expression-based feature extraction for ML model. |
| `09_feature_extraction_essentiality_based.py` | NA | Essentiality-based feature extraction for ML model. |
| `10_feature_merge.py` | NA | Merging all features for machine learning–ready datasets. |
| `11_ML.py` | Fig. 4, Fig. 5, Fig. 6A, Fig. S5-S6 | Machine learning model training and performance evaluation. |

The empirical top-k ranking analysis (Fig. 6B) is implemented per validation screen via `lib/topk.py`; run `validation/lloyd/06_run_topk.py` and `validation/knoll/05_run_topk.py`.

## Validation screen analysis

Two independent CRISPR resistance screens not seen during training are evaluated against the trained RF model using the shared `lib/` modules. Each validation folder contains screen-specific configuration plus thin runner scripts that invoke shared library functions — no duplicated feature extraction or evaluation code.

| Folder | SL pair | Drug | Cell context | Reference |
|--------|---------|------|--------------|-----------|
| `validation/lloyd/` | ATM–ATR | AZD6738 (ATRi) | Mouse mESCs (ATM WT vs KO) | [Lloyd et al., 2021](https://academic.oup.com/nar/article/49/15/8665/6331679) |
| `validation/knoll/` | MTAP–PRMT5 | MRTX1719 / MRTX9768 (MTA-cooperative PRMT5i) | NSCLC (LU99, SW1573) | [Knoll et al., 2025](https://aacrjournals.org/cancerres/article/85/18/3518/764451/CRISPR-Drug-Combinatorial-Screening-Identifies) |

| Script | Lloyd | Knoll | Brief description |
|--------|-------|-------|-------------------|
| `01_extract_labels.py` | ✓ | ✓ | Screen-specific raw-data parsing → top-N labels (Lloyd: MAGeCK Excel; Knoll: AvgDiff CSV). |
| `02_mouse_to_human.py` | ✓ | — | Mouse → human ortholog mapping via Ensembl BioMart + HGNC fallback. |
| `02_format_for_ml.py` | — | ✓ | Format labels into 30-column ML schema using HGNC lookup. |
| `03_format_for_ml.py` | ✓ | — | Format labels into 30-column ML schema using HGNC lookup. |
| `03_run_features.py` / `04_run_features.py` | 04 | 03 | PPI + expression + essentiality + merge — calls `lib.feature_extraction_*` and `lib.merge_features`. |
| `04_run_eval.py` / `05_run_eval.py` | 05 | 04 | RF train + ROC/PR (combined + separate) + bootstrap CI — calls `lib.rf_eval.run_screen_eval`. |
| `05_run_topk.py` / `06_run_topk.py` | 06 | 05 | Top-k prioritization Monte Carlo — calls `lib.topk.run_screen_topk`. |
| `06_run_panel.py` / `07_run_panel.py` | 07 | 06 | Side-by-side panel: ROC curve (left) + top-k (right). Requires eval + topk JSON first. |

## Clinical trial dataset analysis

Clinical-trial biomarker–target validation pipeline applied to PRMT5–MTAP inhibitor candidates.

| Script | Brief description |
|--------|-------------------|
| `clinical_trials/01_ML_data_preprocessing.py` | Preprocess clinical-trial biomarker/target gene table. |
| `clinical_trials/02_feature_extraction_ppi_based.py` | Extract PPI features for trial pairs. |
| `clinical_trials/03_feature_extraction_expression_based.py` | Extract expression features. |
| `clinical_trials/04_feature_extraction_essentiality_based.py` | Extract essentiality features. |
| `clinical_trials/05_feature_merge.py` | Merge into ML-ready dataset. |
| `clinical_trials/06_ML.py` | RF prediction for trial pairs. |

## Shared library (`lib/`)

Single source-of-truth implementation of feature extraction, RF evaluation, and top-k simulation. Imported by both validation screens — no per-screen duplication.

| Module | Purpose |
|--------|---------|
| `lib/hgnc_lookup.py` | Human gene symbol → HGNC/Entrez/Ensembl resolution (symbol → alias → prev_symbol fallbacks). |
| `lib/feature_extraction_ppi.py` | STRING + BIOGRID + Cancer Gene Census PPI features. |
| `lib/feature_extraction_expression.py` | GTEx co-expression + variance + mean expression. |
| `lib/feature_extraction_essentiality.py` | DepMap co-essentiality + percentage. |
| `lib/merge_features.py` | Merge all feature outputs → dropna + dedup. |
| `lib/rf_eval.py` | RandomForest + ROC/PR + bootstrap CI + auto-leakage-exclusion variant. |
| `lib/topk.py` | Top-k Monte Carlo simulation (global negative sampling). |

## Execution Order

Run scripts in the order below. Main pipeline is strict (each step reads prior output). Validation and clinical-trials pipelines depend on the main pipeline finishing through `11_ML.py` (which produces the training matrix consumed downstream); they are independent of each other and can be run in any order or in parallel.

> **Starting point — resistance screen files:** The main pipeline begins with `01_resistance_screen_data_preprocessing.py`, which reads the processed screen files from `input_data/1_resistance_screens/`. These files are already included in the repository (see [Pre-generated Data Files](#pre-generated-data-files)) and can be used directly without any additional steps. Alternatively, if you wish to reproduce the screen files from scratch, each screen has a dedicated raw data analysis script under `input_data/0_raw_data_analysis/<screen>/`. Download the corresponding supplementary files listed in the [Data Sources](#data-sources) section, place them in the appropriate folder, and run the script to regenerate the processed output.

### 1. Main pipeline

```bash
cd predicting_resistance_SL_therapies

python 01_resistance_screen_data_preprocessing.py
python 02_STRING_interaction_overlap_analysis.py
python 03_BIOGRID_interaction_overlap_analysis.py
python 04_interaction_overlap_graphs.py
python 05_randomness_analysis.py
python 06_ML_data_preprocessing.py
python 07_feature_extraction_ppi_based.py
python 08_feature_extraction_expression_based.py
python 09_feature_extraction_essentiality_based.py
python 10_feature_merge.py
python 11_ML.py
```

### 2. Lloyd validation (ATM–ATR, AZD6738)

```bash
cd validation/lloyd
python 01_extract_labels.py
python 02_mouse_to_human.py
python 03_format_for_ml.py
python 04_run_features.py
python 05_run_eval.py
python 06_run_topk.py
python 07_run_panel.py
cd ../..
```

### 3. Knoll validation (MTAP–PRMT5, MRTX1719/MRTX9768)

```bash
cd validation/knoll
python 01_extract_labels.py
python 02_format_for_ml.py
python 03_run_features.py
python 04_run_eval.py
python 05_run_topk.py
python 06_run_panel.py
cd ../..
```

### 4. Clinical trials

```bash
cd clinical_trials
python 01_ML_data_preprocessing.py
python 02_feature_extraction_ppi_based.py
python 03_feature_extraction_expression_based.py
python 04_feature_extraction_essentiality_based.py
python 05_feature_merge.py
python 06_ML.py
cd ..
```

### Dependency notes

- Main `01_` → `11_`: strict order. Each step reads previous output.
- Each validation screen: strict numbered order `01_` → … → `07_` (Lloyd) or `06_` (Knoll). `run_panel.py` requires both `run_eval.py` and `run_topk.py` to have completed.
- Lloyd ⊥ Knoll ⊥ clinical_trials → can run in parallel after main `11_ML.py`.

## Data Sources

### Reference databases

| Category | Path | File | Description | Reference |
|----------|------|------|-------------|-----------|
| Gene annotation | `input_data/HGNC` | `hgnc_complete_set.txt` | Gene symbol → HGNC/Ensembl/Entrez mapping (downloaded April 12, 2024) | [HGNC](https://www.genenames.org/download/custom/) |
| Protein-protein interaction | `input_data/STRING` | `9606.protein.info.v12.0.txt` | STRING protein display names and descriptions (v12.0, December 7, 2023) | [STRING](https://string-db.org/cgi/download?sessionId=bANI3NDYl4wQ) |
| Protein-protein interaction | `input_data/STRING` | `9606.protein.links.detailed.v12.0.txt` | STRING full network with per-channel subscores (v12.0, December 7, 2023) | [STRING](https://string-db.org/cgi/download?sessionId=bANI3NDYl4wQ) |
| Protein-protein interaction | `input_data/BIOGRID` | `BIOGRID-ALL-4.4.241.tab3.txt` | BioGRID all interactions (v4.4.241, November 28, 2024) | [BioGRID](https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.241/) |
| Gene annotation | `input_data/Cancer_Gene_Census` | `Cancer_gene_census_data.csv` | Expert-curated catalogue of genes with causal roles in cancer | [CGC](https://cancer.sanger.ac.uk/cosmic/census) |
| Gene expression | `input_data/GTEx` | `GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz` | GTEx bulk tissue gene expression profiles (v10, February 5, 2025) | [GTEx](https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression) |
| Gene essentiality | `input_data/DepMap` | `CRISPRGeneEffect.csv` | Genome-scale CRISPR dependency profiles (DepMap 23Q4) | [DepMap](https://depmap.org/portal/data_page/?tab=allData) |
| Mouse-human orthology | `input_data/Ensembl` | `mouse_human_orthologs_ensembl.tsv` | Mouse → human ortholog table from Ensembl BioMart (cached locally) | [Ensembl BioMart](https://mart.ensembl.org/biomart/martview) |

### CRISPR screens in training dataset

All screen files are in `input_data/1_resistance_screens/` in `[Class, Gene]` format.

| File | SL Pair | Reference |
|------|---------|-----------|
| `dunn_PTEN_AKT_screen.xlsx` | PTEN–AKT | [Dunn et al., 2022](https://www.nature.com/articles/s41388-022-02482-9) (Supp. Table 1) |
| `dunn_PTEN_PIK3CB_screen.xlsx` | PTEN–PIK3CB | [Dunn et al., 2022](https://www.nature.com/articles/s41388-022-02482-9) (Supp. Table 1) |
| `hayes_NRAS_CDK4_6_screen.xlsx` | NRAS–CDK4/6 | [Hayes et al., 2019](https://aacrjournals.org/cancerres/article/79/9/2352/640672) (Supp. Table 1) |
| `hayes_NRAS_MEK_screen.xlsx` | NRAS–MEK1/2 | [Hayes et al., 2019](https://aacrjournals.org/cancerres/article/79/9/2352/640672) (Supp. Table 1) |
| `krall_BRAF_MEK_screen.xlsx` | BRAF–MEK1/2 | [Krall et al., 2017](https://elifesciences.org/articles/18970) (Supp. File 1) |
| `krall_KRAS_MEK_screen.xlsx` | KRAS–MEK1/2 | [Krall et al., 2017](https://elifesciences.org/articles/18970) (Supp. File 1) |
| `krall_NRAS_MEK_screen.xlsx` | NRAS–MEK1/2 | [Krall et al., 2017](https://elifesciences.org/articles/18970) (Supp. File 1) |
| `llorca_ARID1A_ATR_screen.xlsx` | ARID1A–ATR | [Llorca-Cardenosa et al., 2022](https://aacrjournals.org/cancerres/article/82/21/3962/709958) (Supp. Table 3) |
| `noordermeer_BRCA1_PARP1_screen.xlsx` | BRCA1–PARP1 | [Noordermeer et al., 2018](https://www.nature.com/articles/s41586-018-0340-7) (Supp. Table 1) |
| `dev_BRCA1_PARP1_screen.xlsx` | BRCA1–PARP1 | [Dev et al., 2018](https://www.nature.com/articles/s41556-018-0140-1) (Supp. Table 1) |
| `wang_KRAS_MEK_screen.xlsx` | KRAS–MEK1/2 | [Wang et al., 2017](https://www.sciencedirect.com/science/article/pii/S2211124717300682) (Supp. Table 1) |
| `clements_BRCA2_PARP1_screen.xlsx` | BRCA2–PARP1 | [Clements et al., 2020](https://www.nature.com/articles/s41467-020-19961-w) (Supp. Data-2) |
| `gallo_CCNE1_PKMYT1_screen.xlsx` | CCNE1–PKMYT1 | [Gallo et al., 2022](https://www.nature.com/articles/s41586-022-04638-9) (Supp. Table 1) |

### Validation CRISPR resistance screens

| File | Path | SL Pair | Description | Reference |
|------|------|---------|-------------|-----------|
| `Supplementary Table 1.xlsx` | `validation/lloyd/raw_data` | ATM–ATR | Genome-wide CRISPR screen in ATM-WT/KO mouse mESCs with AZD6738 (ATRi); IC10 and IC90 doses, SUM and REP MAGeCK variants | [Lloyd et al., 2021](https://academic.oup.com/nar/article/49/15/8665/6331679) (Supp. Table 1) |
| `knoll_raw_data.csv` | `validation/knoll/raw_data` | MTAP–PRMT5 | Paralog + single-gene CRISPR screen in NSCLC (LU99, SW1573) with MRTX1719/MRTX9768; LFC differences per cell line + AvgDiff | [Knoll et al., 2025](https://aacrjournals.org/cancerres/article/85/18/3518/764451/CRISPR-Drug-Combinatorial-Screening-Identifies) (Supp. Tables S2, S3) |

## Pre-generated Data Files

These files are committed to the repository and do not need to be generated or downloaded separately. They serve as direct inputs to downstream pipeline steps.

### Processed CRISPR resistance screens (`input_data/1_resistance_screens/`)

Each file contains genome-wide gene-level resistance annotations derived from published CRISPR screens. Format: two columns — `Class` (Resistance / Non-Resistance) and `Gene` (HGNC-approved symbol). Resistance genes are those enriched under drug selection in the original screen; non-resistance genes are the remainder of the screened library. These files are the primary input to `01_resistance_screen_data_preprocessing.py`, which combines them into a unified training dataset for the machine learning model. Each file was produced by a reproducible script in `input_data/0_raw_data_analysis/<screen>/`.

### Clinical trials biomarker–target table (`clinical_trials/biomarker_target_genes.xlsx`)

Defines the 11 synthetic-lethal biomarker–target pairs evaluated in the clinical trials pipeline (e.g., MSH6/WRN, CCNE1/PKMYT1, BRCA1/ATR). Each row specifies a biomarker gene, a primary drug target, and optionally a secondary target. Read by `clinical_trials/01_ML_data_preprocessing.py` as the entry point for the clinical prediction pipeline.

### HGNC-mapped PPI analysis outputs (`input_data/2_outputs_with_hgnc/`)

`analysis_pair_string.csv` is produced by `02_STRING_interaction_overlap_analysis.py` and stores per-gene STRING PPI overlap scores for each SL pair × screen combination with HGNC-resolved symbols, recording whether each gene shares a PPI partner with the biomarker or target in the STRING network (medium confidence ≥ 400). It is consumed by `04_interaction_overlap_graphs.py` and `05_randomness_analysis.py`.

`analysis_pair_biogrid.csv` is the equivalent file built from BioGRID physical interactions, produced by `03_BIOGRID_interaction_overlap_analysis.py` and used by the same downstream scripts to produce interaction enrichment figures and random network comparisons.
