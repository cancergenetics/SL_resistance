#!/usr/bin/env python
# coding: utf-8

# # STRING Interaction Overlap Analysis
# 
# This notebook analyzes protein-protein interaction (PPI) overlap between STRING database interactions and screening data.
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
STRING_LINKS_PATH = Path('./input_data/STRING/9606.protein.links.detailed.v12.0.txt')
STRING_INFO_PATH = Path('./input_data/STRING/9606.protein.info.v12.0.txt')
INPUT_DIR = Path('./input_data/2_outputs_with_hgnc')
OUTPUT_RESULTS_DIR = Path('./results/2_interaction_overlap/string_results')
OUTPUT_COUNTS_DIR = Path('./results/2_interaction_overlap/string_counts')

# Analysis configuration CSV
CONFIG_PATH = Path('./input_data/2_outputs_with_hgnc/analysis_pair_string.csv')

# Ensure output directories exist
OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_COUNTS_DIR.mkdir(parents=True, exist_ok=True)

# STRING score threshold
MIN_COMBINED_SCORE = 400


# ## Load Reference Data (Once)


def load_reference_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load HGNC and STRING reference data.
    Returns STRING interactions, STRING info, and HGNC gene lookup dictionary.
    """
    print("Loading HGNC data...")
    hgnc = pd.read_csv(
        HGNC_PATH, 
        sep='\t',
        usecols=['hgnc_id', 'symbol', 'prev_symbol', 'ensembl_gene_id', 'alias_symbol', 'entrez_id'],
        low_memory=False
    )
    
    print("Loading STRING data...")
    string = pd.read_csv(STRING_LINKS_PATH, sep=' ')
    string_info = pd.read_csv(STRING_INFO_PATH, sep='\t')
    
    # Remove species prefix from protein IDs
    string['protein1'] = string['protein1'].str.replace('9606.', '', regex=False)
    string['protein2'] = string['protein2'].str.replace('9606.', '', regex=False)
    string_info['#string_protein_id'] = string_info['#string_protein_id'].str.replace('9606.', '', regex=False)
    
    # Keep only needed columns from string_info
    string_info = string_info[['#string_protein_id', 'preferred_name']]

    # Build the shared HGNC lookup over every STRING preferred_name (all PPI partner
    # symbols are preferred_names), using lib.hgnc_lookup so it matches the rest of
    # the pipeline. Returns a DataFrame indexed by gene -> hgnc_id/entrez/ensembl.
    print("Building gene lookup (lib.hgnc_lookup, 3-tier)...")
    partner_symbols = string_info['preferred_name'].dropna().unique()
    gene_lookup = build_hgnc_lookup(partner_symbols, HGNC_PATH)

    print(f"Loaded {len(string):,} STRING interactions")
    print(f"Loaded {len(gene_lookup):,} gene symbol mappings")

    return string, string_info, gene_lookup


# Load data
STRING, STRING_INFO, GENE_LOOKUP = load_reference_data()


# ## Core Analysis Functions


def get_ppi_partners(
    protein_ids: List[str],
    string_df: pd.DataFrame,
    string_info: pd.DataFrame,
    gene_lookup: Dict,
    min_score: int = MIN_COMBINED_SCORE
) -> pd.DataFrame:
    """
    Get all PPI partners for the given protein IDs from STRING.
    
    Parameters
    ----------
    protein_ids : list of str
        Ensembl protein IDs (e.g., 'ENSP00000256078')
    string_df : DataFrame
        STRING interactions table
    string_info : DataFrame
        STRING protein info table
    gene_lookup : dict
        Gene symbol to (hgnc_id, ensembl_id, entrez_id) mapping
    min_score : int
        Minimum combined score threshold
    
    Returns
    -------
    DataFrame with interaction partners and gene annotations
    """
    # Collect edges for all query proteins
    edge_dfs = []
    for pid in protein_ids:
        edges = string_df[
            (string_df['protein1'] == pid) | (string_df['protein2'] == pid)
        ]
        edge_dfs.append(edges)
    
    ppi = pd.concat(edge_dfs, ignore_index=True)
    
    # Filter by score
    ppi = ppi[ppi['combined_score'] >= min_score].copy()
    ppi = ppi.sort_values('combined_score', ascending=False)
    
    # Add gene symbols
    ppi = ppi.merge(
        string_info, 
        left_on='protein1', 
        right_on='#string_protein_id', 
        how='left'
    ).rename(columns={'preferred_name': 'gene_symbol_1'}).drop('#string_protein_id', axis=1)
    
    ppi = ppi.merge(
        string_info, 
        left_on='protein2', 
        right_on='#string_protein_id', 
        how='left'
    ).rename(columns={'preferred_name': 'gene_symbol_2'}).drop('#string_protein_id', axis=1)
    
    # Keep only gene symbol columns and deduplicate
    ppi = ppi[['gene_symbol_1', 'gene_symbol_2']].copy()
    
    # Create sorted interaction key for deduplication (A-B == B-A)
    ppi['sorted_pair'] = ppi.apply(
        lambda row: '-'.join(sorted([str(row['gene_symbol_1']), str(row['gene_symbol_2'])])), 
        axis=1
    )
    ppi = ppi.drop_duplicates(subset='sorted_pair').drop(columns='sorted_pair').reset_index(drop=True)
    
    # Add HGNC annotations (gene_lookup is a DataFrame indexed by preferred_name)
    def lookup(symbol):
        if isinstance(symbol, str) and symbol in gene_lookup.index:
            r = gene_lookup.loc[symbol]
            return (r["hgnc_id"], r["ensembl_gene_id"], r["entrez_id"])
        return (np.nan, np.nan, "NA")
    
    ppi[['hgnc_id_1', 'ensembl_gene_id_1', 'entrez_id_1']] = pd.DataFrame(
        ppi['gene_symbol_1'].apply(lookup).tolist(), index=ppi.index
    )
    ppi[['hgnc_id_2', 'ensembl_gene_id_2', 'entrez_id_2']] = pd.DataFrame(
        ppi['gene_symbol_2'].apply(lookup).tolist(), index=ppi.index
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
        Screen data with 'ensembl_gene_id' column
    ppi_partners : DataFrame
        PPI partners from get_ppi_partners()
    
    Returns
    -------
    DataFrame with 'Interaction_Partners' column added
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
    protein_ids: List[str],
    pair_name: str,
    screen_prefix: str,
    input_filename: Optional[str] = None,
    save_outputs: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete interaction overlap analysis.
    
    Counting is always done by the 'Class' column (Resistance / Non-Resistance).
    
    Parameters
    ----------
    protein_ids : list of str
        Ensembl protein IDs for the query proteins
    pair_name : str
        Name for the protein pair/group (used in output filenames)
    screen_prefix : str
        Screen identifier (used in filenames)
    input_filename : str, optional
        Custom input filename. If None, defaults to
        '{screen_prefix}_{pair_name}_screen_with_hgnc.csv'
    save_outputs : bool
        Whether to save output files
    
    Returns
    -------
    ppi_partners : DataFrame
        PPI partners from STRING
    counts : DataFrame
        Interaction counts by Class (Resistance / Non-Resistance)
    """
    print(f"\n{'='*60}")
    print(f"Running analysis: {pair_name} ({screen_prefix})")
    print(f"Proteins: {protein_ids}")
    print(f"{'='*60}")
    
    # Get PPI partners
    ppi_partners = get_ppi_partners(protein_ids, STRING, STRING_INFO, GENE_LOOKUP)
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
        
        results_path = OUTPUT_RESULTS_DIR / f"{output_base}_string_results.csv"
        counts_path = OUTPUT_COUNTS_DIR / f"{output_base}_string_count.csv"
        
        screen.to_csv(results_path, index=False)
        counts.to_csv(counts_path, index=False)
        
        print(f"\nSaved outputs to:")
        print(f"  - {results_path}")
        print(f"  - {counts_path}")
    
    return ppi_partners, screen, counts


# ## Run Analyses
# 
# All analyses are defined in `analysis_config_string.csv` with columns:
# 
# | Column | Description | Example |
# |--------|-------------|---------|
# | `protein_ids` | Semicolon-separated Ensembl protein IDs | `ENSP00000501150;ENSP00000361021` |
# | `pair_name` | Label for the protein pair/group | `PTEN_PIK3CB` |
# | `screen_prefix` | Screen identifier | `dunn` |
# | `input_filename` | Custom filename (leave empty for default `{prefix}_{pair}_screen_with_hgnc.csv`) | `ARID1A_ATR_Awwad_Llorca_aggregated.csv` |
# 
# To add a new analysis, simply add a row to the CSV. To skip one, delete the row.


# Load analysis configuration
config = pd.read_csv(CONFIG_PATH)

print(f"Loaded {len(config)} analyses from {CONFIG_PATH}")



# Run all analyses from config
results = {}

for idx, row in config.iterrows():
    protein_ids = row['protein_ids'].split(';')
    pair_name = row['pair_name']
    screen_prefix = row['screen_prefix']
    input_filename = row['input_filename'] if pd.notna(row['input_filename']) else None
    
    # Build a unique key for storing results
    key = f"{screen_prefix}_{pair_name}"
    
    try:
        ppi, screen, counts = run_interaction_analysis(
            protein_ids=protein_ids,
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
# - `results/2_interaction_overlap/string_results/` — Annotated screen data
# - `results/2_interaction_overlap/string_counts/` — Interaction count files