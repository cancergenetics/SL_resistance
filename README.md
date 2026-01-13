# Predicting Resistance to Synthetic Lethal (SL) Therapies in Cancer	

Data processing and analysis code for the paper of the same name, published in : Genome Medicine

Cite as : []

## External Data Sources

| Folder                         | File                          | Description                                                                 | Source |
|--------------------------------|-------------------------------|-----------------------------------------------------------------------------|--------|
| `input_data/HGNC`              | `hgnc_complete_set.txt`       | Gene symbol mapping to HGNC, Ensembl, and Entrez identifiers                | [HGNC](https://www.genenames.org/download/custom/) |
| `input_data/1_resistance_screens` | `awwad_ARID1A_ATR_screen.xlsx` | CRISPR screen results with gene symbols and resistance status from Awwad et al. (2025) | [Awwad et al. (2025)](https://www.nature.com/articles/s41467-024-55637-5) (Supp. Data-1) |
| `input_data/1_resistance_screens` | `dunn_PTEN_AKT_screen.xlsx` | CRISPR screen results with gene symbols and resistance status for PTEN-AKT SL pair from Dunn et al. (2022) | [Dunn et al. (2022)](https://www.nature.com/articles/s41388-022-02482-9) (Supp. Table-1) |
| `input_data/1_resistance_screens` | `dunn_PTEN__PIK3CB_screen.xlsx` | CRISPR screen results with gene symbols and resistance status for PTEN-PIK3CB SL pair from Dunn et al. (2022) | [Dunn et al. (2022)](https://www.nature.com/articles/s41388-022-02482-9) (Supp. Table-1) |
| `input_data/1_resistance_screens` | `hayes_NRAS_CDK4_6_screen.xlsx` | CRISPR screen results with gene symbols and resistance status for NRAS-CDK4/6 SL pair from Hayes et al. (2019) | [Hayes et al. (2019)](https://aacrjournals.org/cancerres/article/79/9/2352/640672/A-Functional-Landscape-of-Resistance-to-MEK1-2-and) (Supp. Table-1) |
| `input_data/1_resistance_screens` | `hayes_NRAS_MEK_screen.xlsx` | CRISPR screen results with gene symbols and resistance status for NRAS-MEK1/2 SL pair from Hayes et al. (2019) | [Hayes et al. (2019)](https://aacrjournals.org/cancerres/article/79/9/2352/640672/A-Functional-Landscape-of-Resistance-to-MEK1-2-and) (Supp. Table-1) |
| `input_data/1_resistance_screens` | `krall_BRAF_MEK_screen.xlsx` | CRISPR screen results with gene symbols and resistance status for BRAF-MEK1/2 SL pair from Krall et al. (2017) | [Krall et al. (2017)](https://elifesciences.org/articles/18970) (Supp. File-1) |


### Data processing notebooks overview:
These notebooks process raw/third party data for use in the analysis.

| Notebook                               | Brief description                                        |
|:---------------------------------------|:---------------------------------------------------------|
| 1_resistance_screen_data_preprocessing     | Preprocessing, mapping, and annotation of gene symbols in CRISPR resistance screens using HGNC IDs, Entrez IDs, and Ensembl IDs. |

