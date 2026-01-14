# Predicting Resistance to Synthetic Lethal (SL) Therapies in Cancer	

Data processing and analysis code for the paper of the same name, published in : Genome Medicine

Cite as : []

## Data Sources

| Category | Path | File | SL Pair | Description | Reference |
|---------|------|------|--------|-------------|-----------|
| Gene annotation | `input_data/HGNC` | `hgnc_complete_set.txt` | — | Mapping of gene symbols to HGNC, Ensembl, and Entrez identifiers | [HGNC](https://www.genenames.org/download/custom/) |
| Protein-protein interaction | `input_data/STRING` | `9606.protein.info.v12.0.txt` | — | list of STRING proteins incl. their display names and descriptions (downloaded on December 7, 2023, version 12.0) | [STRING](https://string-db.org/cgi/download?sessionId=bANI3NDYl4wQ) |
| Protein-protein interaction | `input_data/STRING` | `9606.protein.links.detailed.v12.0.txt` | — | STRING protein network data full network, incl. subscores per channel (downloaded on December 7, 2023, version 12.0) | [STRING](https://string-db.org/cgi/download?sessionId=bANI3NDYl4wQ) |
| Protein-protein interaction | `input_data/BIOGRID` | `BIOGRID-MV-Physical-4.4.229.tab3.txt` | — | BIOGRID Physical protein interaction network data (downloaded on December 29, 2023, version 4.4.229) | [BIOGRID](https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.229/) |
| CRISPR screen | `input_data/1_resistance_screens` | `awwad_ARID1A_ATR_screen.xlsx` | ARID1A–ATR | CRISPR resistance screen with gene-level resistance annotations | [Awwad et al., 2025](https://www.nature.com/articles/s41467-024-55637-5) (Supp. Data 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `dunn_PTEN_AKT_screen.xlsx` | PTEN–AKT | CRISPR resistance screen with gene-level resistance annotations | [Dunn et al., 2022](https://www.nature.com/articles/s41388-022-02482-9) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `dunn_PTEN_PIK3CB_screen.xlsx` | PTEN–PIK3CB | CRISPR resistance screen with gene-level resistance annotations | [Dunn et al., 2022](https://www.nature.com/articles/s41388-022-02482-9) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `hayes_NRAS_CDK4_6_screen.xlsx` | NRAS–CDK4/6 | CRISPR resistance screen with gene-level resistance annotations | [Hayes et al., 2019](https://aacrjournals.org/cancerres/article/79/9/2352/640672) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `hayes_NRAS_MEK_screen.xlsx` | NRAS–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Hayes et al., 2019](https://aacrjournals.org/cancerres/article/79/9/2352/640672) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `krall_BRAF_MEK_screen.xlsx` | BRAF–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Krall et al., 2017](https://elifesciences.org/articles/18970) (Supp. File 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `krall_KRAS_MEK_screen.xlsx` | KRAS–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Krall et al., 2017](https://elifesciences.org/articles/18970) (Supp. File 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `krall_NRAS_MEK_screen.xlsx` | NRAS–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Krall et al., 2017](https://elifesciences.org/articles/18970) (Supp. File 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `llorca_ARID1A_ATR_screen.xlsx` | ARID1A–ATR | CRISPR resistance screen with gene-level resistance annotations | [Llorca-Cardenosa et al., 2022](https://aacrjournals.org/cancerres/article/82/21/3962/709958) (Supp. Data 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `noordermeer_BRCA1_PARP1_screen.xlsx` | BRCA1–PARP1 | CRISPR resistance screen with gene-level resistance annotations | [Noordermeer et al., 2018](https://www.nature.com/articles/s41586-018-0340-7) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `wang_KRAS_MEK_screen.xlsx` | KRAS–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Wang et al., 2017](https://www.sciencedirect.com/science/article/pii/S2211124717300682) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `yu_KRAS_MEK_screen.xlsx` | KRAS–MEK1/2 | CRISPR resistance screen with gene-level resistance annotations | [Yu et al., 2022](https://www.nature.com/articles/s41388-021-02077-w) (Supp. Table 1) |
| CRISPR screen | `input_data/1_resistance_screens` | `zimmermann_BRCA1_PARP1_screen.xlsx` | BRCA1–PARP1 | CRISPR resistance screen with gene-level resistance annotations | [Zimmermann et al., 2018](https://www.nature.com/articles/s41586-018-0291-z) (Supp. Table 1) |




## Data processing notebooks overview:
These notebooks process raw/third party data for use in the analysis.

| Notebook                                  | Figures | Brief description |
|-------------------------------------------|---------|-------------------|
| `1_resistance_screen_data_preprocessing.ipynb`  | Fig. S1 | Preprocessing, mapping, and annotation of gene symbols in CRISPR resistance screens using HGNC, Entrez, and Ensembl identifiers. |
| `2_STRING_interaction_overlap_analysis.ipynb`  | NA | Overlapping analysis of protein-protein interaction partners of SL pairs and resistance/non-resistance genes in STRING (medium confidence) database |
| `3_BIOGRID_interaction_overlap_analysis.ipynb`  | NA | Overlapping analysis of protein-protein interaction partners of SL pairs and resistance/non-resistance genes in BIOGRID physical database |
| `4_interaction_overlap_graphs.ipynb`  | Fig. 1B, Fig.2, Fig.S2-S3 | Visualization of overlapping analysis of protein-protein interaction partner enrichment of SL pairs among resistance/non-resistance genes across CRISPR screens |
