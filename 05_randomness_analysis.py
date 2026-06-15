#!/usr/bin/env python
# coding: utf-8

# # Randomness Analysis — Degree-Preserving Random Networks
# 
# This notebook:
# 1. **Generates** 1,000 degree-preserving random networks for STRING and BioGRID (configuration model)
# 2. **Runs** interaction overlap analysis against each random network for every screen
# 3. **Merges** per-screen random results into a single file
# 4. **Plots** histograms comparing the real network to the random distribution
# 


import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# Single shared HGNC resolver (strict 3-tier symbol > prev > alias, case-insensitive).
from lib.hgnc_lookup import build_hgnc_lookup

np.random.seed(42)


# ## Configuration


# File paths
HGNC_PATH = Path('input_data/HGNC/hgnc_complete_set.txt')
STRING_LINKS_PATH = Path('input_data/STRING/9606.protein.links.detailed.v12.0.txt')
STRING_INFO_PATH = Path('input_data/STRING/9606.protein.info.v12.0.txt')
BIOGRID_PATH = Path('input_data/BIOGRID/BIOGRID-ALL-4.4.241.tab3.txt')
INPUT_DIR = Path('input_data/2_outputs_with_hgnc')

STRING_RANDOM_DIR = Path('input_data/4_randomness/string_random_networks')
BIOGRID_RANDOM_DIR = Path('input_data/4_randomness/biogrid_random_networks')
OUTPUT_DIR = Path('results/4_randomness')

# Create output directories
STRING_RANDOM_DIR.mkdir(parents=True, exist_ok=True)
BIOGRID_RANDOM_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
MIN_COMBINED_SCORE = 400
N_RANDOM = 1000


# ## Load Reference Data (Once)


# === HGNC ===
# The shared gene lookup is built AFTER STRING + BioGRID load (over their partner
# symbols), using lib.hgnc_lookup — see the block right after the BioGRID load below.



# === STRING ===
print("Loading STRING data...")
STRING = pd.read_csv(STRING_LINKS_PATH, sep=' ')
STRING_INFO = pd.read_csv(STRING_INFO_PATH, sep='\t')

STRING['protein1'] = STRING['protein1'].str.replace('9606.', '', regex=False)
STRING['protein2'] = STRING['protein2'].str.replace('9606.', '', regex=False)
STRING_INFO['#string_protein_id'] = STRING_INFO['#string_protein_id'].str.replace('9606.', '', regex=False)
STRING_INFO = STRING_INFO[['#string_protein_id', 'preferred_name']]

# Deduplicate and filter
STRING_FILTERED = STRING[STRING['combined_score'] >= MIN_COMBINED_SCORE].copy()
STRING_FILTERED['sorted_pair'] = STRING_FILTERED.apply(
    lambda r: '-'.join(sorted([r['protein1'], r['protein2']])), axis=1)
STRING_FILTERED = STRING_FILTERED.drop_duplicates(subset='sorted_pair', keep='first')
STRING_FILTERED = STRING_FILTERED.drop(columns='sorted_pair').reset_index(drop=True)

print(f"Loaded {len(STRING_FILTERED):,} STRING interactions (score >= {MIN_COMBINED_SCORE})")



# === BIOGRID ===
print("Loading BioGRID data...")
BIOGRID = pd.read_csv(BIOGRID_PATH, sep='\t', low_memory=False)
BIOGRID = BIOGRID[
    (BIOGRID['Experimental System Type'] == 'physical') &
    (BIOGRID['Organism Name Interactor A'] == 'Homo sapiens') &
    (BIOGRID['Organism Name Interactor B'] == 'Homo sapiens')
].copy()

# Deduplicate
BIOGRID['sorted_pair'] = BIOGRID.apply(
    lambda r: '-'.join(sorted([str(r['Official Symbol Interactor A']),
                                str(r['Official Symbol Interactor B'])])), axis=1)
BIOGRID = BIOGRID.drop_duplicates(subset='sorted_pair', keep='first')
BIOGRID = BIOGRID.drop(columns='sorted_pair').reset_index(drop=True)

print(f"Loaded {len(BIOGRID):,} BioGRID human interactions")


# === Shared HGNC lookup (lib.hgnc_lookup; strict 3-tier, case-insensitive) ===
# Built over every partner symbol that can appear: STRING preferred_names + BioGRID
# official symbols. Same resolver as 01/02/03 so mappings are identical pipeline-wide.
_partner_symbols = pd.unique(pd.concat([
    STRING_INFO['preferred_name'],
    BIOGRID['Official Symbol Interactor A'],
    BIOGRID['Official Symbol Interactor B'],
], ignore_index=True).dropna())
GENE_LOOKUP = build_hgnc_lookup(_partner_symbols, HGNC_PATH)
print(f"Built {len(GENE_LOOKUP):,} gene symbol mappings (lib.hgnc_lookup)")

def lookup_gene_info(gene_symbol):
    if isinstance(gene_symbol, str) and gene_symbol in GENE_LOOKUP.index:
        r = GENE_LOOKUP.loc[gene_symbol]
        return (r["hgnc_id"], r["ensembl_gene_id"], r["entrez_id"])
    return (np.nan, np.nan, "NA")


# ## Part 1 — Generate 1,000 Random Networks

# This cell can be skipped if the 1,000 random networks have already been generated. You can continue from Part-2


def generate_random_networks(edge_df, col_a, col_b, output_dir, prefix, n_random=N_RANDOM):
    """Generate degree-preserving random networks using the configuration model.

    Parameters
    ----------
    edge_df : DataFrame with interaction columns
    col_a, col_b : column names for the two interactor IDs
    output_dir : directory to write random network files
    prefix : filename prefix (e.g. 'string_random' or 'biogrid_random')
    n_random : number of random networks to generate
    """
    edge_list = list(zip(edge_df[col_a], edge_df[col_b]))
    G = nx.Graph()
    G.add_edges_from(edge_list)
    G.remove_edges_from(nx.selfloop_edges(G))  # remove self-loops before degree extraction

    node_ids = list(G.nodes())
    node_degrees = [G.degree(n) for n in node_ids]

    print(f"Network: {len(node_ids):,} nodes, {G.number_of_edges():,} edges")
    print(f"Generating {n_random} random networks...")

    for i in range(n_random):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_random}")
        # Configuration model: exact degree sequence preserved.
        # Returns MultiGraph (parallel edges + self-loops possible) → convert to simple graph.
        r = nx.configuration_model(node_degrees, seed=42 + i)
        r = nx.Graph(r)  # removes parallel edges
        r.remove_edges_from(nx.selfloop_edges(r))
        out_path = os.path.join(output_dir, f'{prefix}_{i}.txt')
        with open(out_path, 'w') as f:
            for e in r.edges:
                f.write(f"{node_ids[e[0]]}\t{node_ids[e[1]]}\n")

    print(f"Done. Files saved to {output_dir}/")


def _maybe_generate(edge_df, col_a, col_b, output_dir, prefix, n_random=N_RANDOM):
    existing = sum(1 for i in range(n_random) if os.path.exists(os.path.join(output_dir, f'{prefix}_{i}.txt')))
    if existing == n_random:
        print(f"Skipping {prefix}: all {n_random} files already exist in {output_dir}/")
        return
    print(f"Found {existing}/{n_random} existing files — regenerating all.")
    generate_random_networks(edge_df, col_a, col_b, output_dir, prefix, n_random)


# Generate STRING random networks
_maybe_generate(
    STRING_FILTERED, 'protein1', 'protein2',
    STRING_RANDOM_DIR, 'string_random', N_RANDOM
)


# Generate BioGRID random networks
_maybe_generate(
    BIOGRID, 'Official Symbol Interactor A', 'Official Symbol Interactor B',
    BIOGRID_RANDOM_DIR, 'biogrid_random', N_RANDOM
)


# ## Part 2 — STRING: Random Network Interaction Analysis


def get_string_partners(protein_ids, string_df):
    """Get PPI partner ensembl IDs for a list of Ensembl protein IDs from a STRING edge table."""
    edge_dfs = []
    for pid in protein_ids:
        edges = string_df[
            (string_df['protein1'] == pid) | (string_df['protein2'] == pid)
        ]
        edge_dfs.append(edges)
    ppi = pd.concat(edge_dfs, ignore_index=True)

    # Merge with STRING info to get gene names → ensembl IDs via HGNC
    ppi = ppi.merge(STRING_INFO, left_on='protein1', right_on='#string_protein_id', how='left')
    ppi = ppi.rename(columns={'preferred_name': 'gene_symbol_1'}).drop(columns='#string_protein_id', errors='ignore')
    ppi = ppi.merge(STRING_INFO, left_on='protein2', right_on='#string_protein_id', how='left')
    ppi = ppi.rename(columns={'preferred_name': 'gene_symbol_2'}).drop(columns='#string_protein_id', errors='ignore')

    # Deduplicate
    ppi['sorted_pair'] = ppi.apply(lambda r: '-'.join(sorted([str(r['protein1']), str(r['protein2'])])), axis=1)
    ppi = ppi.drop_duplicates(subset='sorted_pair', keep='first').drop(columns='sorted_pair')

    # HGNC lookup
    ppi[['hgnc_id_1', 'ensembl_gene_id_1', 'entrez_id_1']] = pd.DataFrame(
        ppi['gene_symbol_1'].apply(lookup_gene_info).tolist(), index=ppi.index)
    ppi[['hgnc_id_2', 'ensembl_gene_id_2', 'entrez_id_2']] = pd.DataFrame(
        ppi['gene_symbol_2'].apply(lookup_gene_info).tolist(), index=ppi.index)

    return ppi


def run_string_random_analysis(
    protein_ids: List[str],
    pair_name: str,
    screen_prefix: str,
    input_filename: Optional[str] = None,
    n_random: int = N_RANDOM,
):
    """Run real + N random STRING network analyses for a set of query proteins.

    Parameters
    ----------
    protein_ids : list of Ensembl protein IDs (e.g. ['ENSP00000361021', 'ENSP00000501150'])
    pair_name : label for the pair (e.g. 'PTEN_PIK3CB')
    screen_prefix : screen identifier (e.g. 'dunn')
    input_filename : custom input filename, or None for auto
    n_random : number of random networks
    """
    print(f"\n{'='*60}")
    print(f"STRING random analysis: {pair_name} ({screen_prefix})")
    print(f"Proteins: {protein_ids}")
    print(f"{'='*60}")

    # Real network
    real_partners = get_string_partners(protein_ids, STRING_FILTERED)

    # Load screen
    if input_filename is None:
        input_filename = f'{screen_prefix}_{pair_name}_screen_with_hgnc.csv'
    screen = pd.read_csv(INPUT_DIR / input_filename)
    valid_ids = screen['ensembl_gene_id'].fillna('').astype(str).str.strip()

    # Real interaction column
    partner_ids_real = (
        set(real_partners['ensembl_gene_id_1'].dropna().astype(str)) |
        set(real_partners['ensembl_gene_id_2'].dropna().astype(str))
    )
    screen['Interaction_REAL'] = ((valid_ids != '') & valid_ids.isin(partner_ids_real)).astype(int)
    print(f"Real network: {screen['Interaction_REAL'].sum()} partners")

    # Random networks
    random_cols = {}
    for i in range(1, n_random + 1):
        if i % 100 == 0:
            print(f"  Random network {i}/{n_random}...")
        file_path = STRING_RANDOM_DIR / f'string_random_{i-1}.txt'
        if not file_path.exists():
            print(f"  ⚠️ Missing: {file_path}")
            continue

        rand_df = pd.read_csv(file_path, sep='\t', header=None, names=['protein1', 'protein2'])
        rand_df['protein1'] = rand_df['protein1'].astype(str).str.replace('9606.', '', regex=False)
        rand_df['protein2'] = rand_df['protein2'].astype(str).str.replace('9606.', '', regex=False)

        rand_partners = get_string_partners(protein_ids, rand_df)
        rand_ids = (
            set(rand_partners['ensembl_gene_id_1'].dropna().astype(str)) |
            set(rand_partners['ensembl_gene_id_2'].dropna().astype(str))
        )
        random_cols[f'Interaction_Random_{i}'] = ((valid_ids != '') & valid_ids.isin(rand_ids)).astype(int)

    screen = pd.concat([screen, pd.DataFrame(random_cols)], axis=1)
    print(f"✅ Done. {len(random_cols)} random networks processed.")
    return screen



# Load STRING analysis config (same CSV used overlap analysis)
STRING_CONFIG_PATH = Path('input_data/2_outputs_with_hgnc/analysis_pair_string.csv')
string_config = pd.read_csv(STRING_CONFIG_PATH)
print(f"Loaded {len(string_config)} STRING analyses")



for _, row in string_config.iterrows():
    protein_ids = row['protein_ids'].split(';')
    input_fn = row['input_filename'] if pd.notna(row['input_filename']) else None
    output_name = f"{row['pair_name']}_random_network_string_results.csv"

    screen_out = run_string_random_analysis(
        protein_ids=protein_ids,
        pair_name=row['pair_name'],
        screen_prefix=row['screen_prefix'],
        input_filename=input_fn,
    )
    screen_out.to_csv(OUTPUT_DIR / output_name, index=False)
    print(f"Saved: {OUTPUT_DIR / output_name}")


# ## Part 3 — Merge STRING Random Results & Plot Histogram


def merge_random_results(folder, pattern_suffix, output_filename, class_column='Class'):
    """Merge per-screen random network result files into one combined file.

    Uses the 'Class' column to create a binary Resistance indicator:
        Resistance → 1, Non-Resistance → 0
    """
    from glob import glob

    file_list = sorted(glob(os.path.join(folder, f'*{pattern_suffix}')))
    print(f"Found {len(file_list)} files to merge.")

    merged_list = []
    for file_path in file_list:
        df = pd.read_csv(file_path)
        base_name = os.path.basename(file_path)
        pair_name = base_name.replace(pattern_suffix, '')

        parts = pair_name.split('_', 1)
        biomarker = parts[0]
        target = parts[1] if len(parts) > 1 else None

        df['Pair_Name'] = pair_name
        df['Biomarker'] = biomarker
        df['Target'] = target

        # Map Class to binary Resistance column
        if class_column in df.columns:
            class_lower = df[class_column].str.strip().str.lower()
            df['Resistance'] = 0
            df.loc[class_lower.str.contains('resistance') & ~class_lower.str.contains('non'), 'Resistance'] = 1

        # Rename Gene → Query if present
        if 'Gene' in df.columns:
            df = df.rename(columns={'Gene': 'Query'})

        # Drop ID columns
        df = df.drop(columns=[c for c in ['hgnc_id', 'ensembl_gene_id', 'entrez_id'] if c in df.columns])

        # Reorder
        first_cols = ['Query', 'Biomarker', 'Target', 'Resistance']
        first_cols = [c for c in first_cols if c in df.columns]
        rest = [c for c in df.columns if c not in first_cols]
        df = df[first_cols + rest]

        merged_list.append(df)
        print(f"  ✅ {pair_name}: {len(df)} rows")

    merged_df = pd.concat(merged_list, ignore_index=True)
    out_path = os.path.join(folder, output_filename)
    merged_df.to_csv(out_path, index=False)
    print(f"\nMerged {len(file_list)} files → {out_path} ({len(merged_df):,} rows)")
    return merged_df



# Merge STRING random results
merged_string = merge_random_results(
    str(OUTPUT_DIR),
    '_random_network_string_results.csv',
    'all_pairs_merged_random_network_results_clean.csv'
)



def plot_random_histogram(merged_df, real_label, save_path,
                         figsize=(10, 6), xlabel_fontsize=13, dpi=600):
    """Plot histogram of random network interaction sums vs real network.

    Uses Okabe-Ito colour palette for accessibility.
    """
    OI_ORANGE = "#E69F00"
    OI_PURPLE = "#CC79A7"
    OI_BLACK = "#000000"

    random_cols = [c for c in merged_df.columns if c.startswith('Interaction_Random_')]
    res_df = merged_df[merged_df['Resistance'] == 1]

    random_sums = res_df[random_cols].sum(axis=0).values
    real_network = res_df['Interaction_REAL'].sum()

    sorted_desc = np.sort(random_sums)[::-1]
    val_10 = sorted_desc[9] if len(sorted_desc) >= 10 else None    # p = 0.01
    val_50 = sorted_desc[49] if len(sorted_desc) >= 50 else None   # p = 0.05

    plt.figure(figsize=figsize)
    plt.hist(random_sums, bins=20, edgecolor='black', linewidth=0)
    plt.xlabel("number of resistance genes connected to biomarker or target", fontsize=xlabel_fontsize)

    if val_10 is not None:
        plt.axvline(val_10, color=OI_ORANGE, linestyle='dashed', linewidth=2, label="p < 0.01", zorder=2)
    if val_50 is not None:
        plt.axvline(val_50, color=OI_PURPLE, linestyle='dashed', linewidth=4, label="p < 0.05", zorder=3)
    plt.axvline(real_network, color=OI_BLACK, linestyle='solid', linewidth=2,
                label=f"Real {real_label} network", zorder=4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.show()
    print(f"Saved: {save_path}")



# STRING histogram
merged_string = pd.read_csv(OUTPUT_DIR / 'all_pairs_merged_random_network_results_clean.csv')
plot_random_histogram(
    merged_string, 'STRING',
    OUTPUT_DIR / 'random_networks_string_interaction_histogram.jpeg'
)


# ## Part 4 — BioGRID: Random Network Interaction Analysis


def get_biogrid_partners(protein_names, biogrid_df):
    """Get PPI partner ensembl IDs for a list of gene symbols from a BioGRID edge table."""
    col_a = 'Official Symbol Interactor A'
    col_b = 'Official Symbol Interactor B'

    edge_dfs = []
    for name in protein_names:
        edges = biogrid_df[(biogrid_df[col_a] == name) | (biogrid_df[col_b] == name)]
        edge_dfs.append(edges)
    ppi = pd.concat(edge_dfs, ignore_index=True)

    # Deduplicate
    ppi['sorted_pair'] = ppi.apply(
        lambda r: '-'.join(sorted([str(r[col_a]), str(r[col_b])])), axis=1)
    ppi = ppi.drop_duplicates(subset='sorted_pair', keep='first').drop(columns='sorted_pair')

    # HGNC lookup
    ppi[['hgnc_id_1', 'ensembl_gene_id_1', 'entrez_id_1']] = pd.DataFrame(
        ppi[col_a].apply(lookup_gene_info).tolist(), index=ppi.index)
    ppi[['hgnc_id_2', 'ensembl_gene_id_2', 'entrez_id_2']] = pd.DataFrame(
        ppi[col_b].apply(lookup_gene_info).tolist(), index=ppi.index)

    return ppi


def run_biogrid_random_analysis(
    protein_names: List[str],
    pair_name: str,
    screen_prefix: str,
    input_filename: Optional[str] = None,
    n_random: int = N_RANDOM,
):
    """Run real + N random BioGRID network analyses for a set of query gene symbols."""
    print(f"\n{'='*60}")
    print(f"BioGRID random analysis: {pair_name} ({screen_prefix})")
    print(f"Genes: {protein_names}")
    print(f"{'='*60}")

    # Real network
    real_partners = get_biogrid_partners(protein_names, BIOGRID)

    # Load screen
    if input_filename is None:
        input_filename = f'{screen_prefix}_{pair_name}_screen_with_hgnc.csv'
    screen = pd.read_csv(INPUT_DIR / input_filename)
    valid_ids = screen['ensembl_gene_id'].fillna('').astype(str).str.strip()

    # Real interaction column
    partner_ids_real = (
        set(real_partners['ensembl_gene_id_1'].dropna().astype(str)) |
        set(real_partners['ensembl_gene_id_2'].dropna().astype(str))
    )
    screen['Interaction_REAL'] = ((valid_ids != '') & valid_ids.isin(partner_ids_real)).astype(int)
    print(f"Real network: {screen['Interaction_REAL'].sum()} partners")

    # Random networks
    random_cols = {}
    for i in range(1, n_random + 1):
        if i % 100 == 0:
            print(f"  Random network {i}/{n_random}...")
        file_path = BIOGRID_RANDOM_DIR / f'biogrid_random_{i-1}.txt'
        if not file_path.exists():
            print(f"  ⚠️ Missing: {file_path}")
            continue

        rand_df = pd.read_csv(file_path, sep='\t', header=None,
                              names=['Official Symbol Interactor A', 'Official Symbol Interactor B'])

        rand_partners = get_biogrid_partners(protein_names, rand_df)
        rand_ids = (
            set(rand_partners['ensembl_gene_id_1'].dropna().astype(str)) |
            set(rand_partners['ensembl_gene_id_2'].dropna().astype(str))
        )
        random_cols[f'Interaction_Random_{i}'] = ((valid_ids != '') & valid_ids.isin(rand_ids)).astype(int)

    screen = pd.concat([screen, pd.DataFrame(random_cols)], axis=1)
    print(f"✅ Done. {len(random_cols)} random networks processed.")
    return screen



# Load BioGRID analysis config (same CSV used in the overlap analysis)
BIOGRID_CONFIG_PATH = Path('input_data/2_outputs_with_hgnc/analysis_pair_biogrid.csv')
biogrid_config = pd.read_csv(BIOGRID_CONFIG_PATH)
print(f"Loaded {len(biogrid_config)} BioGRID analyses")
print(biogrid_config.to_string(index=False))



for _, row in biogrid_config.iterrows():
    protein_names = row['protein_names'].split(';')
    input_fn = row['input_filename'] if pd.notna(row['input_filename']) else None
    output_name = f"{row['pair_name']}_random_network_biogrid_results.csv"

    screen_out = run_biogrid_random_analysis(
        protein_names=protein_names,
        pair_name=row['pair_name'],
        screen_prefix=row['screen_prefix'],
        input_filename=input_fn,
    )
    screen_out.to_csv(OUTPUT_DIR / output_name, index=False)
    print(f"Saved: {OUTPUT_DIR / output_name}")


# ## Part 5 — Merge BioGRID Random Results & Plot Histogram


# Merge BioGRID random results
merged_biogrid = merge_random_results(
    str(OUTPUT_DIR),
    '_random_network_biogrid_results.csv',
    'all_pairs_merged_random_network_results_clean_biogrid.csv'
)



# BioGRID histogram
merged_biogrid = pd.read_csv(OUTPUT_DIR / 'all_pairs_merged_random_network_results_clean_biogrid.csv')
plot_random_histogram(
    merged_biogrid, 'BIOGRID Physical',
    OUTPUT_DIR / 'random_networks_biogrid_interaction_histogram.jpeg'
)


# ## Summary
# 
# All outputs saved to `results/4_randomness/`:
# - Per-screen result files with `Interaction_REAL` + `Interaction_Random_1..1000` columns
# - Merged result files for STRING and BioGRID
# - Histogram figures comparing real vs random network distributions

# 