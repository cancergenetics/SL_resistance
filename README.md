# Predicting Resistance to Synthetic Lethal (SL) Therapies in Cancer

This repository contains Python scripts and files for data processing and analysis code for the paper of the same name, published in : Genome Medicine

Cite as : []

## Data processing scripts overview

These scripts process raw / third-party data and reproduce all main analyses. Notebooks were converted to `.py` for reproducibility.

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

## Validation screens

Two independent CRISPR resistance screens not seen during training are evaluated against the trained RF model using the shared `lib/` modules. Each validation folder contains screen-specific configuration plus thin runner scripts that invoke shared library functions — no duplicated feature extraction or evaluation code.

| Folder | SL pair | Drug | Cell context | Reference |
|--------|---------|------|--------------|-----------|
| `validation/lloyd/` | ATM–ATR | AZD6738 (ATRi) | Mouse mESCs (ATM WT vs KO) | [Lloyd et al., 2021](https://doi.org/10.1016/j.celrep.2021.108797) |
| `validation/knoll/` | MTAP–PRMT5 | MRTX1719 / MRTX9768 (MTA-cooperative PRMT5i) | NSCLC (LU99, SW1573) | [Knoll et al., 2025](https://aacrjournals.org/cancerres/article/85/18/3518) |

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

## Clinical trials scripts

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

## Environment

The code was developed and tested using a Conda environment with **Python 3.12**.

To recreate the environment, run:

```bash
conda env create -f environment.yml
conda activate predicting_resistance_sl
```

## Execution Order

Run scripts in the order below. Main pipeline is strict (each step reads prior output). Validation and clinical-trials pipelines depend on the main pipeline finishing through `11_ML.py` (which produces the training matrix consumed downstream); they are independent of each other and can be run in any order or in parallel.

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

| Category | Path | File | SL Pair | Description | Reference |
|---------|------|------|--------|-------------|-----------|
| Gene annotation | `input_data/HGNC` | `hgnc_complete_set.txt` | — | Mapping of gene symbols to HGNC, Ensembl, and Entrez identifiers (downloaded on April 12, 2024) | [HGNC](https://www.genenames.org/download/custom/) |
| Protein-protein interaction | `input_data/STRING` | `9606.protein.info.v12.0.txt` | — | List of STRING proteins incl. their display names and descriptions (downloaded on December 7, 2023, version 12.0) | [STRING](https://string-db.org/cgi/download?sessionId=bANI3NDYl4wQ) |
| Protein-protein interaction | `input_data/STRING` | `9606.protein.links.detailed.v12.0.txt` | — | STRING protein network data, full network, incl. subscores per channel (downloaded on December 7, 2023, version 12.0) | [STRING](https://string-db.org/cgi/download?sessionId=bANI3NDYl4wQ) |
| Protein-protein interaction | `input_data/BIOGRID` | `BIOGRID-MV-Physical-4.4.229.tab3.txt` | — | BIOGRID Physical protein interaction network data (downloaded on December 29, 2023, version 4.4.229) | [BIOGRID_Physical](https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.229/) |
| Protein-protein interaction | `input_data/BIOGRID` | `BIOGRID-ALL-4.4.241.tab3.txt` | — | BIOGRID protein interaction network data (downloaded on November 28, 2024, version 4.4.241) | [BIOGRID](https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.241/) |
| Gene annotation | `input_data/Cancer_Gene_Census` | `Cancer_gene_census_data.csv` | — | A high-confidence, expert-curated catalogue of genes with causal roles in human cancer | [CGC](https://cancer.sanger.ac.uk/cosmic/census) |
| Gene expression | `input_data/GTEx` | `GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz` | — | Gene expression profiles from Genotype-Tissue Expression (GTEx) bulk tissue data (downloaded on February 5, 2025, version 10, RNASeQCv2.4.2) | [GTEx](https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression) |
| Gene essentiality | `input_data/DepMap` | `CRISPRGeneEffect.csv` | — | Genome-scale CRISPR dependency profiles reported by the Cancer Dependency Map (DepMap, 23Q4 dataset) | [DepMap](https://depmap.org/portal/data_page/?tab=allData) |
| Mouse-human orthology | `input_data/Ensembl` | `mouse_human_orthologs_ensembl.tsv` | — | Mouse → human gene ortholog table (Ensembl BioMart `mmusculus_gene_ensembl`, cached locally) | [Ensembl BioMart](https://mart.ensembl.org/biomart/martview) |
| CRISPR screen | `input_data/1_resistance_screens` | `dunn_PTEN_AKT_screen.xlsx` | PTEN–AKT | CRISPR resistance screen with gene-level resistance annotations | [Dunn et al., 2022](https://www.nature.com/articles/s41388-022-02482-9) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `dunn_PTEN_PIK3CB_screen.xlsx` | PTEN–PIK3CB | CRISPR resistance screen with gene-level resistance annotations | [Dunn et al., 2022](https://www.nature.com/articles/s41388-022-02482-9) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `hayes_NRAS_CDK4_6_screen.xlsx` | NRAS–CDK4/6 | CRISPR resistance screen with gene-level resistance annotations | [Hayes et al., 2019](https://aacrjournals.org/cancerres/article/79/9/2352/640672) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `hayes_NRAS_MEK_screen.xlsx` | NRAS–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Hayes et al., 2019](https://aacrjournals.org/cancerres/article/79/9/2352/640672) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `krall_BRAF_MEK_screen.xlsx` | BRAF–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Krall et al., 2017](https://elifesciences.org/articles/18970) (Supp. File 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `krall_KRAS_MEK_screen.xlsx` | KRAS–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Krall et al., 2017](https://elifesciences.org/articles/18970) (Supp. File 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `krall_NRAS_MEK_screen.xlsx` | NRAS–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Krall et al., 2017](https://elifesciences.org/articles/18970) (Supp. File 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `llorca_ARID1A_ATR_screen.xlsx` | ARID1A–ATR | CRISPR resistance screen with gene-level resistance annotations | [Llorca-Cardenosa et al., 2022](https://aacrjournals.org/cancerres/article/82/21/3962/709958) (Supp. Table 3) |
| CRISPR screen | `input_data/1_resistance_screens` | `noordermeer_BRCA1_PARP1_screen.xlsx` | BRCA1–PARP1 | CRISPR resistance screen with gene-level resistance annotations | [Noordermeer et al., 2018](https://www.nature.com/articles/s41586-018-0340-7) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `dev_BRCA1_PARP1_screen.xlsx` | BRCA1–PARP1 | CRISPR resistance screen with gene-level resistance annotations | [Dev et al., 2018](https://www.nature.com/articles/s41556-018-0140-1) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `wang_KRAS_MEK_screen.xlsx` | KRAS–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Wang et al., 2017](https://www.sciencedirect.com/science/article/pii/S2211124717300682) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `yu_KRAS_MEK_screen.xlsx` | KRAS–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Yu et al., 2022](https://www.nature.com/articles/s41388-021-02077-w) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `clements_BRCA2_PARP1_screen.xlsx` | BRCA2–PARP1 | CRISPR resistance screen with gene-level resistance annotations | [Clements et al., 2020](https://www.nature.com/articles/s41467-020-19961-w) (Supp. Data-2) |
| CRISPR screen | `input_data/1_resistance_screens` | `gallo_CCNE1_PKMYT1_screen.xlsx` | CCNE1–PKMYT1 | CRISPR resistance screen with gene-level resistance annotations | [Gallo et al., 2022](https://www.nature.com/articles/s41586-022-04638-9) (Supp. Table 1) |

## Validation Data Sources

| Category | Path | File | SL Pair | Description | Reference |
|---------|------|------|--------|-------------|-----------|
| Validation screen | `validation/lloyd/raw_data` | `Supplementary Table 1.xlsx` | ATM–ATR | Genome-wide CRISPR resistance screen in ATM-WT and ATM-KO mouse mESCs treated with AZD6738 (ATR inhibitor); IC10 and IC90 doses, SUM and REP MAGeCK analysis variants | [Lloyd et al., 2021](https://www.cell.com/cell-reports/fulltext/S2211-1247(21)00217-6) (Supp. Table 1) |
| Validation screen | `validation/knoll/raw_data` | `knoll_raw_data.csv` | MTAP–PRMT5 | Paralog + single-gene CRISPR screen in NSCLC cell lines (LU99, SW1573) treated with MTA-cooperative PRMT5 inhibitors (MRTX1719, MRTX9768); LFC differences (MRTXi vs DMSO) per cell line + AvgDiff | [Knoll et al., 2025](https://aacrjournals.org/cancerres/article/85/18/3518) (Supp. Tables S2, S3) |
