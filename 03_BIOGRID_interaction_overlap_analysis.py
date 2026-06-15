#!/usr/bin/env python
# coding: utf-8

# # BioGRID Interaction Overlap Analysis
# 
# This notebook analyzes protein-protein interaction (PPI) overlap between BioGRID database interactions and screening data.
# 


import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings

# Single shared HGNC resolver (strict 3-tier symbol > prev > alias, case-insensitive).
from lib.hgnc_lookup import build_hgnc_lookup

# Optional visualization imports (uncomment if needed)
# import matplotlib.pyplot as plt
# from matplotlib_venn import venn2, venn3
# from scipy.stats import fisher_exact


# ## Configuration


# File paths - adjust these to your environment
HGNC_PATH = Path('./input_data/HGNC/hgnc_complete_set.txt')
BIOGRID_PATH = Path('./input_data/BIOGRID/BIOGRID-ALL-4.4.241.tab3.txt')
INPUT_DIR = Path('./input_data/2_outputs_with_hgnc')
OUTPUT_RESULTS_DIR = Path('./results/2_interaction_overlap/biogrid_results')
OUTPUT_COUNTS_DIR = Path('./results/2_interaction_overlap/biogrid_counts')

# Analysis configuration CSV
CONFIG_PATH = Path('./input_data/2_outputs_with_hgnc/analysis_pair_biogrid.csv')

# Ensure output directories exist
OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_COUNTS_DIR.mkdir(parents=True, exist_ok=True)


# ## Load Reference Data (Once)


def load_reference_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Load HGNC and BioGRID reference data.
    Returns BioGRID interactions (human only) and HGNC gene lookup dictionary.
    """
    print("Loading HGNC data...")
    hgnc = pd.read_csv(
        HGNC_PATH,
        sep='\t',
        usecols=['hgnc_id', 'symbol', 'prev_symbol', 'ensembl_gene_id', 'alias_symbol', 'entrez_id'],
        low_memory=False
    )

    print("Loading BioGRID data...")
    biogrid = pd.read_csv(BIOGRID_PATH, sep='\t', low_memory=False)

    # Filter to physical human-human interactions only
    biogrid = biogrid[
        (biogrid['Experimental System Type'] == 'physical') &
        (biogrid['Organism Name Interactor A'] == 'Homo sapiens') &
        (biogrid['Organism Name Interactor B'] == 'Homo sapiens')
    ].copy()

    # Build the shared HGNC lookup over every BioGRID official symbol (all PPI partner
    # symbols), using lib.hgnc_lookup so it matches the rest of the pipeline. Returns a
    # DataFrame indexed by gene -> hgnc_id/entrez/ensembl.
    print("Building gene lookup (lib.hgnc_lookup, 3-tier)...")
    partner_symbols = pd.unique(pd.concat([
        biogrid['Official Symbol Interactor A'],
        biogrid['Official Symbol Interactor B'],
    ], ignore_index=True).dropna())
    gene_lookup = build_hgnc_lookup(partner_symbols, HGNC_PATH)

    print(f"Loaded {len(biogrid):,} BioGRID human interactions")
    print(f"Loaded {len(gene_lookup):,} gene symbol mappings")

    return biogrid, gene_lookup


# Load data
BIOGRID, GENE_LOOKUP = load_reference_data()


# ## Core Analysis Functions


def get_ppi_partners(
    protein_names: List[str],
    biogrid_df: pd.DataFrame,
    gene_lookup: Dict
) -> pd.DataFrame:
    """
    Get all PPI partners for the given gene symbols from BioGRID.

    Parameters
    ----------
    protein_names : list of str
        Gene symbols to query (e.g., ['PTEN', 'PIK3CB']).
        Can include aliases (e.g., ['MAP2K1', 'MEK1']) — all are searched.
    biogrid_df : DataFrame
        BioGRID interactions table (already filtered to human).
    gene_lookup : dict
        Gene symbol to (hgnc_id, ensembl_id, entrez_id) mapping.

    Returns
    -------
    DataFrame with deduplicated interaction partners and HGNC annotations.
    """
    col_a = 'Official Symbol Interactor A'
    col_b = 'Official Symbol Interactor B'

    # Collect edges for all query gene names
    edge_dfs = []
    for name in protein_names:
        edges = biogrid_df[
            (biogrid_df[col_a] == name) | (biogrid_df[col_b] == name)
        ]
        edge_dfs.append(edges)

    ppi = pd.concat(edge_dfs, ignore_index=True)

    # Deduplicate undirected edges (A-B == B-A)
    ppi['sorted_pair'] = ppi.apply(
        lambda row: '-'.join(sorted([str(row[col_a]), str(row[col_b])])),
        axis=1
    )
    ppi = ppi.drop_duplicates(subset='sorted_pair', keep='first').drop(columns='sorted_pair')
    ppi = ppi.reset_index(drop=True)

    # Add HGNC annotations (gene_lookup is a DataFrame indexed by official symbol)
    def lookup(symbol):
        if isinstance(symbol, str) and symbol in gene_lookup.index:
            r = gene_lookup.loc[symbol]
            return (r["hgnc_id"], r["ensembl_gene_id"], r["entrez_id"])
        return (np.nan, np.nan, "NA")

    ppi[['hgnc_id_1', 'ensembl_gene_id_1', 'entrez_id_1']] = pd.DataFrame(
        ppi[col_a].apply(lookup).tolist(), index=ppi.index
    )
    ppi[['hgnc_id_2', 'ensembl_gene_id_2', 'entrez_id_2']] = pd.DataFrame(
        ppi[col_b].apply(lookup).tolist(), index=ppi.index
    )

    return ppi


def annotate_screen_with_interactions(
    screen: pd.DataFrame,
    ppi_partners: pd.DataFrame
) -> pd.DataFrame:
    """
    Annotate screen data with interaction partner flags.

    Parameters
    ----------
    screen : DataFrame
        Screen data with 'ensembl_gene_id' column.
    ppi_partners : DataFrame
        PPI partners from get_ppi_partners().

    Returns
    -------
    DataFrame with 'Interaction_Partners' column added.
    """
    screen = screen.copy()

    # Normalize screen IDs
    valid_ids = screen['ensembl_gene_id'].fillna('').astype(str).str.strip()

    # Get all partner ensembl IDs
    partner_ids_1 = set(ppi_partners['ensembl_gene_id_1'].dropna().astype(str))
    partner_ids_2 = set(ppi_partners['ensembl_gene_id_2'].dropna().astype(str))
    all_partner_ids = partner_ids_1 | partner_ids_2

    # Flag interactions
    screen['Interaction_Partners'] = (
        (valid_ids != '') & valid_ids.isin(all_partner_ids)
    ).astype(int)

    return screen



def count_interactions_by_class(screen: pd.DataFrame) -> pd.DataFrame:
    """
    Count interactions grouped by the 'Class' column.

    Expects the 'Class' column to contain values such as 'Resistance' and
    'Non-Resistance'.  All unique values in the column are counted.

    Parameters
    ----------
    screen : DataFrame
        Screen data with 'Class' and 'Interaction_Partners' columns.

    Returns
    -------
    DataFrame with columns: Class, Interacted, Total
    """
    assert 'Class' in screen.columns, (
        f"Screen data is missing the 'Class' column. "
        f"Available columns: {list(screen.columns)}"
    )
    assert 'Interaction_Partners' in screen.columns, (
        "Screen data is missing the 'Interaction_Partners' column. "
        "Run annotate_screen_with_interactions() first."
    )

    # Check for unexpected NaNs
    n_missing = screen['Class'].isna().sum()
    if n_missing > 0:
        warnings.warn(
            f"'Class' column has {n_missing} missing values — "
            f"these rows will be excluded from counts."
        )

    counts = (
        screen
        .dropna(subset=['Class'])
        .groupby('Class', sort=False)['Interaction_Partners']
        .agg(Interacted='sum', Total='count')
        .reset_index()
    )
    counts['Interacted'] = counts['Interacted'].astype(int)

    # Reorder so Resistance comes first, Non-Resistance second
    class_order = ['Resistance', 'Non-Resistance']
    present_order = [c for c in class_order if c in counts['Class'].values]
    remaining = [c for c in counts['Class'].values if c not in class_order]
    ordered = present_order + remaining
    counts['Class'] = pd.Categorical(counts['Class'], categories=ordered, ordered=True)
    counts = counts.sort_values('Class').reset_index(drop=True)

    return counts



def run_interaction_analysis(
    protein_names: List[str],
    pair_name: str,
    screen_prefix: str,
    input_filename: Optional[str] = None,
    save_outputs: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete BioGRID interaction overlap analysis.

    Counting is always done by the 'Class' column (Resistance / Non-Resistance).

    Parameters
    ----------
    protein_names : list of str
        Gene symbols for the query proteins (including aliases).
    pair_name : str
        Name for the protein pair/group (used in output filenames).
    screen_prefix : str
        Screen identifier (used in filenames).
    input_filename : str, optional
        Custom input filename. If None, defaults to
        '{screen_prefix}_{pair_name}_screen_with_hgnc.csv'
    save_outputs : bool
        Whether to save output files.

    Returns
    -------
    ppi_partners : DataFrame
        PPI partners from STRING
    counts : DataFrame
        Interaction counts by Class (Resistance / Non-Resistance).
    """
    print(f"\n{'='*60}")
    print(f"Running analysis: {pair_name} ({screen_prefix})")
    print(f"Gene symbols: {protein_names}")
    print(f"{'='*60}")

    # Get PPI partners
    ppi_partners = get_ppi_partners(protein_names, BIOGRID, GENE_LOOKUP)
    print(f"Found {len(ppi_partners)} unique interaction pairs")

    # Load screen data
    if input_filename is None:
        input_filename = f"{screen_prefix}_{pair_name}_screen_with_hgnc.csv"
    screen_path = INPUT_DIR / input_filename

    if not screen_path.exists():
        raise FileNotFoundError(f"Screen file not found: {screen_path}")

    screen = pd.read_csv(screen_path)
    print(f"Loaded screen with {len(screen)} genes")

    # Annotate with interactions
    screen = annotate_screen_with_interactions(screen, ppi_partners)
    n_partners = screen['Interaction_Partners'].sum()
    print(f"Found {n_partners} genes that are interaction partners ({100*n_partners/len(screen):.1f}%)")

    # Count interactions by Class column
    counts = count_interactions_by_class(screen)

    # Validate totals
    n_counted = counts['Total'].sum()
    n_class_notna = screen['Class'].notna().sum()
    assert n_counted == n_class_notna, (
        f"Count totals ({n_counted}) != non-null Class rows ({n_class_notna}). "
        f"Check 'Class' column values."
    )

    print("\nInteraction counts:")
    print(counts.to_string(index=False))

    # Convert binary to labels for output
    screen['Interaction_Partners'] = screen['Interaction_Partners'].map(
        {0: 'Not partner', 1: 'Partner'}
    )

    # Save outputs
    if save_outputs:
        if input_filename and 'aggregated' in input_filename:
            output_base = f"{pair_name}_{screen_prefix}_aggregated"
        else:
            output_base = f"{screen_prefix}_{pair_name}"

        results_path = OUTPUT_RESULTS_DIR / f"{output_base}_biogrid_results.csv"
        counts_path = OUTPUT_COUNTS_DIR / f"{output_base}_biogrid_count.csv"

        screen.to_csv(results_path, index=False)
        counts.to_csv(counts_path, index=False)

        print(f"\nSaved outputs to:")
        print(f"  - {results_path}")
        print(f"  - {counts_path}")

    return ppi_partners, screen, counts


# ## Run Analyses
# 
# All analyses are defined in `biogrid_analysis_config.csv` with columns:
# 
# | Column | Description | Example |
# |--------|-------------|---------|
# | `protein_names` | Semicolon-separated gene symbols (including aliases) | `KRAS;MAP2K1;MEK1;MAP2K2;MEK2` |
# | `pair_name` | Label for the protein pair/group | `KRAS_MEK` |
# | `screen_prefix` | Screen identifier | `wang` |
# | `input_filename` | Custom filename (leave empty for default `{prefix}_{pair}_screen_with_hgnc.csv`) | `KRAS_MEK_Krall_Wang_Yu_aggregated.csv` |
# 
# To add a new analysis, simply add a row to the CSV. To skip one, delete the row.


# Load analysis configuration
config = pd.read_csv(CONFIG_PATH)

print(f"Loaded {len(config)} analyses from {CONFIG_PATH}")
print()
print(config.to_string(index=False))



# Run all analyses from config
results = {}

for idx, row in config.iterrows():
    protein_names = row['protein_names'].split(';')
    pair_name = row['pair_name']
    screen_prefix = row['screen_prefix']
    input_filename = row['input_filename'] if pd.notna(row['input_filename']) else None

    # Build a unique key for storing results
    key = f"{screen_prefix}_{pair_name}"

    try:
        ppi, screen, counts = run_interaction_analysis(
            protein_names=protein_names,
            pair_name=pair_name,
            screen_prefix=screen_prefix,
            input_filename=input_filename,
        )
        results[key] = {
            'ppi_partners': ppi,
            'screen': screen,
            'counts': counts,
        }
    except FileNotFoundError as e:
        print(f"\n*** SKIPPED {key}: {e}")
    except Exception as e:
        print(f"\n*** FAILED {key}: {type(e).__name__}: {e}")
        raise

print(f"\n{'='*60}")
print(f"Completed: {len(results)} / {len(config)} analyses")
print(f"{'='*60}")


# All analyses complete. Output files are saved to:
# - `results/2_interaction_overlap/biogrid_results/` — Annotated screen data
# - `results/2_interaction_overlap/biogrid_counts/` — Interaction count summaries