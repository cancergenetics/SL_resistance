#!/usr/bin/env python
# coding: utf-8

# # Interaction Overlap Graphs
# 
# This notebook generates all figures for the interaction overlap analysis.
# 
# ## Structure
# 1. **Helper functions** — statistics, plotting, and file loading (defined once)
# 2. **Figure 1B** — Single pair example (PTEN-PIK3CB)
# 3. **Figure 2A** — Screen-by-screen analysis (STRING & BioGRID)
# 4. **Figure 2B** — Pooled analysis across all screens
# 5. **Supplementary figures** — Subgroup analyses (TSG-only, excluding PARP, excluding MEK)
# 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from pathlib import Path



# Paths
STRING_COUNTS_DIR = Path('results/2_interaction_overlap/string_counts')
BIOGRID_COUNTS_DIR = Path('results/2_interaction_overlap/biogrid_counts')
STRING_RESULTS_DIR = Path('results/2_interaction_overlap/string_results')
BIOGRID_RESULTS_DIR = Path('results/2_interaction_overlap/biogrid_results')
GRAPH_OUTPUT_DIR = Path('results/3_interaction_overlap_graphs')
GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Config CSVs (same files used by notebooks 2, 3, 5, 6)
CONFIG_DIR = Path('input_data/2_outputs_with_hgnc')
STRING_CONFIG_PATH = CONFIG_DIR / 'analysis_pair_string.csv'
BIOGRID_CONFIG_PATH = CONFIG_DIR / 'analysis_pair_biogrid.csv'


# ## Helper Functions


def significance_stars(pval):
    """Return significance stars for a p-value."""
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'n.s.'


def compute_fisher_from_count_file(count_path):
    """Read a count CSV with Class/Interacted/Total columns and compute
    Fisher's exact test (Resistance vs Non-Resistance).

    Returns dict with contingency table values, OR, p-value, and percentages.
    """
    df = pd.read_csv(count_path)

    class_col = df['Class'].str.strip().str.lower()
    res_mask = class_col.str.contains('resistance') & ~class_col.str.contains('non')
    non_res_mask = class_col.str.contains('non')

    res_interacted = df.loc[res_mask, 'Interacted'].sum()
    res_total = df.loc[res_mask, 'Total'].sum()
    non_res_interacted = df.loc[non_res_mask, 'Interacted'].sum()
    non_res_total = df.loc[non_res_mask, 'Total'].sum()

    res_not_interacted = res_total - res_interacted
    non_res_not_interacted = non_res_total - non_res_interacted

    contingency = [
        [res_interacted, non_res_interacted],
        [res_not_interacted, non_res_not_interacted],
    ]
    odds_ratio, p_value = fisher_exact(contingency)

    return {
        'Interacted_Resistant': int(res_interacted),
        'Interacted_Non_Resistant': int(non_res_interacted),
        'Not_Interacted_Resistant': int(res_not_interacted),
        'Not_Interacted_Non_Resistant': int(non_res_not_interacted),
        'Odds_Ratio': odds_ratio,
        'P_Value': p_value,
        'Resistant_ratio': (res_interacted / res_total * 100) if res_total > 0 else 0,
        'Not_Resistant_ratio': (non_res_interacted / non_res_total * 100) if non_res_total > 0 else 0,
    }


def compute_fisher_from_results_files(file_paths):
    """Pool multiple annotated screen result CSVs and compute a single
    Fisher's exact test across the combined data.

    Reads 'Class' and 'Interaction_Partners' columns.
    """
    frames = [pd.read_csv(fp) for fp in file_paths]
    merged = pd.concat(frames, ignore_index=True)

    class_col = merged['Class'].str.strip().str.lower()
    res_mask = class_col.str.contains('resistance') & ~class_col.str.contains('non')
    non_res_mask = class_col.str.contains('non')

    resistant = merged.loc[res_mask]
    non_resistant = merged.loc[non_res_mask]

    res_partner = (resistant['Interaction_Partners'] == 'Partner').sum()
    res_not_partner = (resistant['Interaction_Partners'] == 'Not partner').sum()
    non_res_partner = (non_resistant['Interaction_Partners'] == 'Partner').sum()
    non_res_not_partner = (non_resistant['Interaction_Partners'] == 'Not partner').sum()

    contingency = [
        [res_partner, non_res_partner],
        [res_not_partner, non_res_not_partner],
    ]
    odds_ratio, p_value = fisher_exact(contingency)

    res_total = res_partner + res_not_partner
    non_res_total = non_res_partner + non_res_not_partner

    return {
        'Interacted_Resistant': int(res_partner),
        'Interacted_Non_Resistant': int(non_res_partner),
        'Not_Interacted_Resistant': int(res_not_partner),
        'Not_Interacted_Non_Resistant': int(non_res_not_partner),
        'Odds_Ratio': odds_ratio,
        'P_Value': p_value,
        'Resistant_ratio': (res_partner / res_total * 100) if res_total > 0 else 0,
        'Not_Resistant_ratio': (non_res_partner / non_res_total * 100) if non_res_total > 0 else 0,
    }



def draw_significance_bracket(ax, bar1, bar2, p_value, odds_ratio, fontsize=16):
    """Draw a bracket between two bars with significance stars and OR."""
    height = max(bar1.get_height(), bar2.get_height())
    line_y = height * 1.05
    bracket_dy = height * 0.02

    x1 = bar1.get_x() + bar1.get_width() / 2
    x2 = bar2.get_x() + bar2.get_width() / 2

    ax.plot([x1, x1], [line_y, line_y + bracket_dy], lw=1.5, color='black')
    ax.plot([x2, x2], [line_y, line_y + bracket_dy], lw=1.5, color='black')
    ax.plot([x1, x2], [line_y + bracket_dy, line_y + bracket_dy], lw=1.5, color='black')

    stars = significance_stars(p_value)
    text = f'{stars}\nOR: {odds_ratio:.2f}'
    ax.annotate(
        text,
        xy=((x1 + x2) / 2, line_y + bracket_dy),
        xytext=(0, 5),
        textcoords='offset points',
        ha='center', va='bottom',
        fontsize=fontsize,
    )


def plot_paired_bars(
    results_df, x_labels, ylabel, save_path,
    figsize=(16, 12), ylim_max=30,
    bar_width=0.4, bar_linewidth=2.5,
    color_non_res=None, color_res=None,
    ylabel_fontsize=30, xtick_fontsize=28, ytick_fontsize=22,
    annotation_fontsize=20, legend_fontsize=24,
    x_rotation=0, dpi=600,
):
    """Plot paired bar chart (Non-Resistance vs Resistance) with significance brackets.

    All font/size parameters default to the Figure 2B values from the original notebook.
    Override them per-figure to match original formatting exactly.
    """
    positions = range(len(results_df))

    fig, ax = plt.subplots(figsize=figsize)

    bar_kwargs_non_res = dict(
        width=bar_width, label='Non-Resistance',
        edgecolor='black', linewidth=bar_linewidth,
    )
    bar_kwargs_res = dict(
        width=bar_width, label='Resistance',
        edgecolor='black', linewidth=bar_linewidth,
    )
    if color_non_res:
        bar_kwargs_non_res['color'] = color_non_res
    if color_res:
        bar_kwargs_res['color'] = color_res

    bars1 = ax.bar(
        [p - bar_width / 2 for p in positions],
        results_df['Not_Resistant_ratio'],
        **bar_kwargs_non_res,
    )
    bars2 = ax.bar(
        [p + bar_width / 2 for p in positions],
        results_df['Resistant_ratio'],
        **bar_kwargs_res,
    )

    ax.set_ylim(0, ylim_max)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(x_labels, fontsize=xtick_fontsize, rotation=x_rotation,
                       ha='right' if x_rotation else 'center')
    ax.tick_params(axis='y', labelsize=ytick_fontsize)

    for bar1, bar2, p, or_val in zip(
        bars1, bars2, results_df['P_Value'], results_df['Odds_Ratio'],
    ):
        draw_significance_bracket(ax, bar1, bar2, p, or_val, fontsize=annotation_fontsize)

    # For Fig1B style: manual legend with color patches
    if color_non_res and color_res:
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_non_res),
                   plt.Rectangle((0, 0), 1, 1, color=color_res)]
        ax.legend(handles, ['Non-Resistance', 'Resistance'], fontsize=legend_fontsize)
    else:
        ax.legend(fontsize=legend_fontsize)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.show()
    print(f'Saved: {save_path}')


# ---
# ## Figure 1B — Single Pair (PTEN-PIK3CB)


# Original: figsize=(8,8), bar_width=0.35, linewidth=1.5, ylabel fs=24,
#           xtick fs=20, ytick fs=20, annotation fs=16, legend fs=18,
#           colors: blue / darkorange, ylim=auto+20, dpi=600

string_stats = compute_fisher_from_count_file(
    STRING_COUNTS_DIR / 'dunn_PTEN_PIK3CB_string_count.csv'
)
biogrid_stats = compute_fisher_from_count_file(
    BIOGRID_COUNTS_DIR / 'dunn_PTEN_PIK3CB_biogrid_count.csv'
)

fig1b_df = pd.DataFrame([string_stats, biogrid_stats])
print(fig1b_df[['Resistant_ratio', 'Not_Resistant_ratio', 'Odds_Ratio', 'P_Value']])

plot_paired_bars(
    fig1b_df,
    x_labels=['STRING', 'BioGRID Physical'],
    ylabel='% of protein interaction partners \nof PTEN and PIK3CB',
    save_path=GRAPH_OUTPUT_DIR / 'figure1b_pten_pi3kcb_graph.jpg',
    figsize=(8, 8),
    bar_width=0.35,
    bar_linewidth=1.5,
    color_non_res='blue',
    color_res='darkorange',
    ylabel_fontsize=24,
    xtick_fontsize=20,
    ytick_fontsize=20,
    annotation_fontsize=16,
    legend_fontsize=18,
    ylim_max=max(fig1b_df['Resistant_ratio'].max(),
                 fig1b_df['Not_Resistant_ratio'].max()) + 20,
)


# ---
# ## Figure 2A — Screen-by-Screen Analysis


# Define all individual screens and their count file names
SCREEN_CONFIG = [
    {'label': 'PTEN-AKT',    'string_count': 'dunn_PTEN_AKT_string_count.csv',
                              'biogrid_count': 'dunn_PTEN_AKT_biogrid_count.csv'},
    {'label': 'PTEN-PIK3CB', 'string_count': 'dunn_PTEN_PIK3CB_string_count.csv',
                              'biogrid_count': 'dunn_PTEN_PIK3CB_biogrid_count.csv'},
    {'label': 'ARID1A-ATR',  'string_count': 'Llorca_ARID1A_ATR_string_count.csv',
                              'biogrid_count': 'Llorca_ARID1A_ATR_biogrid_count.csv'},
    {'label': 'KRAS-MEK',    'string_count': 'KRAS_MEK_Krall_Wang_aggregated_string_count.csv',
                              'biogrid_count': 'KRAS_MEK_Krall_Wang_aggregated_biogrid_count.csv'},
    {'label': 'BRCA1-PARP1', 'string_count': 'BRCA1_PARP1_Dev_Noordermeer_aggregated_string_count.csv',
                              'biogrid_count': 'BRCA1_PARP1_Dev_Noordermeer_aggregated_biogrid_count.csv'},
    {'label': 'BRCA2-PARP1', 'string_count': 'clements_BRCA2_PARP1_string_count.csv',
                              'biogrid_count': 'clements_BRCA2_PARP1_biogrid_count.csv'},
    {'label': 'NRAS-MEK',    'string_count': 'NRAS_MEK_Hayes_Krall_aggregated_string_count.csv',
                              'biogrid_count': 'NRAS_MEK_Hayes_Krall_aggregated_biogrid_count.csv'},
    {'label': 'NRAS-CDK4/6', 'string_count': 'hayes_NRAS_CDK4_6_string_count.csv',
                              'biogrid_count': 'hayes_NRAS_CDK4_6_biogrid_count.csv'},
    {'label': 'BRAF-MEK',    'string_count': 'krall_BRAF_MEK_string_count.csv',
                              'biogrid_count': 'krall_BRAF_MEK_biogrid_count.csv'},
    {'label': 'CCNE1-PKMYT1',    'string_count': 'gallo_CCNE1_PKMYT1_string_count.csv',
                                  'biogrid_count': 'gallo_CCNE1_PKMYT1_biogrid_count.csv'}
]



string_rows = []
for cfg in SCREEN_CONFIG:
    stats = compute_fisher_from_count_file(STRING_COUNTS_DIR / cfg['string_count'])
    stats['Synthetic_Lethal_Partners'] = cfg['label']
    string_rows.append(stats)

string_screen_df = pd.DataFrame(string_rows)
string_screen_df.to_csv(GRAPH_OUTPUT_DIR / 'all_results_seperate_screen_string.csv', index=False)

plot_paired_bars(
    string_screen_df,
    x_labels=string_screen_df['Synthetic_Lethal_Partners'],
    ylabel='% protein interaction partners of \n synthetic lethal pairs in STRING',
    save_path=GRAPH_OUTPUT_DIR / 'figure2a_string_all_screens.jpg',
    figsize=(16, 12),
    bar_width=0.4,
    bar_linewidth=1.5,
    ylabel_fontsize=30,
    xtick_fontsize=28,
    ytick_fontsize=22,
    annotation_fontsize=20,
    legend_fontsize=24,
    ylim_max=40,
    x_rotation=45,
)



# Original: figsize=(16,12), bar_width=0.4, linewidth=1.5, ylabel fs=30,
#           xtick fs=28, ytick fs=22, annotation fs=20, legend fs=24,
#           rotation=45, ylim=20, dpi=600

biogrid_rows = []
for cfg in SCREEN_CONFIG:
    stats = compute_fisher_from_count_file(BIOGRID_COUNTS_DIR / cfg['biogrid_count'])
    stats['Synthetic_Lethal_Partners'] = cfg['label']
    biogrid_rows.append(stats)

biogrid_screen_df = pd.DataFrame(biogrid_rows)
biogrid_screen_df.to_csv(GRAPH_OUTPUT_DIR / 'all_results_seperate_screen_biogrid.csv', index=False)

plot_paired_bars(
    biogrid_screen_df,
    x_labels=biogrid_screen_df['Synthetic_Lethal_Partners'],
    ylabel='% protein interaction partners of \n synthetic lethal pairs in BioGRID Physical',
    save_path=GRAPH_OUTPUT_DIR / 'supplementary_figure_2_biogrid_all_screens.jpg',
    figsize=(16, 12),
    bar_width=0.4,
    bar_linewidth=1.5,
    ylabel_fontsize=30,
    xtick_fontsize=28,
    ytick_fontsize=22,
    annotation_fontsize=20,
    legend_fontsize=24,
    ylim_max=30,
    x_rotation=45,
)


# ---
# ## Figure 2B — Pooled Analysis (All Screens)


# All result files for pooled analysis
ALL_STRING_RESULTS = [
    STRING_RESULTS_DIR / 'dunn_PTEN_PIK3CB_string_results.csv',
    STRING_RESULTS_DIR / 'dunn_PTEN_AKT_string_results.csv',
    STRING_RESULTS_DIR / 'Llorca_ARID1A_ATR_string_results.csv',
    STRING_RESULTS_DIR / 'KRAS_MEK_Krall_Wang_aggregated_string_results.csv',
    STRING_RESULTS_DIR / 'BRCA1_PARP1_Dev_Noordermeer_aggregated_string_results.csv',
    STRING_RESULTS_DIR / 'NRAS_MEK_Hayes_Krall_aggregated_string_results.csv',
    STRING_RESULTS_DIR / 'hayes_NRAS_CDK4_6_string_results.csv',
    STRING_RESULTS_DIR / 'krall_BRAF_MEK_string_results.csv',
    STRING_RESULTS_DIR / 'clements_BRCA2_PARP1_string_results.csv',
    STRING_RESULTS_DIR / 'gallo_CCNE1_PKMYT1_string_results.csv',

]

ALL_BIOGRID_RESULTS = [
    BIOGRID_RESULTS_DIR / 'dunn_PTEN_PIK3CB_biogrid_results.csv',
    BIOGRID_RESULTS_DIR / 'dunn_PTEN_AKT_biogrid_results.csv',
    BIOGRID_RESULTS_DIR / 'Llorca_ARID1A_ATR_biogrid_results.csv',
    BIOGRID_RESULTS_DIR / 'KRAS_MEK_Krall_Wang_aggregated_biogrid_results.csv',
    BIOGRID_RESULTS_DIR / 'BRCA1_PARP1_Dev_Noordermeer_aggregated_biogrid_results.csv',
    BIOGRID_RESULTS_DIR / 'NRAS_MEK_Hayes_Krall_aggregated_biogrid_results.csv',
    BIOGRID_RESULTS_DIR / 'hayes_NRAS_CDK4_6_biogrid_results.csv',
    BIOGRID_RESULTS_DIR / 'krall_BRAF_MEK_biogrid_results.csv',
    BIOGRID_RESULTS_DIR / 'clements_BRCA2_PARP1_biogrid_results.csv',
    BIOGRID_RESULTS_DIR / 'gallo_CCNE1_PKMYT1_biogrid_results.csv',
]



# Original: figsize=(16,12), bar_width=0.4, linewidth=2.5, ylabel fs=30,
#           xtick fs=28, ytick fs=22, annotation fs=26, legend fs=24,
#           ylim=25, dpi=600

pooled_string = compute_fisher_from_results_files(ALL_STRING_RESULTS)
pooled_biogrid = compute_fisher_from_results_files(ALL_BIOGRID_RESULTS)

pooled_df = pd.DataFrame([pooled_string, pooled_biogrid])
pooled_df.to_csv(GRAPH_OUTPUT_DIR / 'total_results_duplicate_allow.csv', index=False)
print(pooled_df[['Resistant_ratio', 'Not_Resistant_ratio', 'Odds_Ratio', 'P_Value']])

plot_paired_bars(
    pooled_df,
    x_labels=['STRING', 'BioGRID Physical'],
    ylabel='% protein interaction partners of \n synthetic lethal pairs',
    save_path=GRAPH_OUTPUT_DIR / 'figure2b_total_sum_table.jpg',
    figsize=(16, 12),
    bar_width=0.4,
    bar_linewidth=2.5,
    ylabel_fontsize=30,
    xtick_fontsize=28,
    ytick_fontsize=22,
    annotation_fontsize=26,
    legend_fontsize=24,
    ylim_max=25,
)


# ---
# ## Supplementary Figures — Subgroup Analyses
# 
# Each subgroup pools a different subset of screens.


# Subgroup definitions — indices refer to ALL_STRING/BIOGRID_RESULTS lists:
#   0: PTEN-PIK3CB    1: PTEN-AKT       2: ARID1A-ATR      3: KRAS-MEK
#   4: BRCA1-PARP1    5: NRAS-MEK       6: NRAS-CDK4/6     7: BRAF-MEK.    8: BRCA2-PARP1 9: CCNE1-PKMYT1
SUBGROUPS = {
    'tsg': {
        'description': 'Tumour suppressor only (PTEN, BRCA1, ARID1A)',
        'indices': [0, 1, 2, 4, 8],
        'ylabel': '% protein interaction partners of\n SL pairs with a tumour suppressor as a biomarker ',
        'filename': 'supplementary_fig_2_tsg_total.jpeg',
    },
    'exc_parp': {
        'description': 'Excluding PARP inhibitor screens',
        'indices': [0, 1, 2, 3, 5, 6, 7, 9],
        'ylabel': '% protein interaction partners',
        'filename': 'supplementary_fig_2_brca_exc_total.jpeg',
    },
    'exc_mek': {
        'description': 'Excluding MEK inhibitor screens',
        'indices': [0, 1, 2, 4, 6, 7, 8, 9],
        'ylabel': '% protein interaction partners',
        'filename': 'supplementary_fig_2_exc_mek.jpeg',
    },
}



# Compute stats for each subgroup
subgroup_dfs = {}

for name, cfg in SUBGROUPS.items():
    idx = cfg['indices']
    string_files = [ALL_STRING_RESULTS[i] for i in idx]
    biogrid_files = [ALL_BIOGRID_RESULTS[i] for i in idx]

    s = compute_fisher_from_results_files(string_files)
    b = compute_fisher_from_results_files(biogrid_files)

    df = pd.DataFrame([s, b])
    subgroup_dfs[name] = df

    print(f"\n{cfg['description']}:")
    print(df[['Resistant_ratio', 'Not_Resistant_ratio', 'Odds_Ratio', 'P_Value']].to_string(index=False))



fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

data_list = [
    (subgroup_dfs['tsg'],
     '% protein interaction partners of \n SL pairs with a tumour suppressor \n as a biomarker'),
    (subgroup_dfs['exc_parp'],
     '% protein interaction partners of \n SL pairs when excluding screens \n involving PARP inhibitors'),
    (subgroup_dfs['exc_mek'],
     '% protein interaction partners of \n SL pairs when excluding screens \n involving MEK inhibitors '),
]

width = 0.4
subplot_labels = ['A', 'B', 'C']

for ax, (data, ylabel), label in zip(axes, data_list, subplot_labels):
    positions = range(len(data))

    bars1 = ax.bar(
        [p - width / 2 for p in positions],
        data['Not_Resistant_ratio'],
        width=width, label='Non-Resistance',
        edgecolor='black', linewidth=2.5,
    )
    bars2 = ax.bar(
        [p + width / 2 for p in positions],
        data['Resistant_ratio'],
        width=width, label='Resistance',
        edgecolor='black', linewidth=2.5,
    )

    ax.set_ylim(0, 30)
    ax.set_ylabel(ylabel, fontsize=26)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(['STRING', 'BioGRID Physical'], fontsize=26)
    ax.tick_params(axis='y', labelsize=26)

    # Larger, bold, centered panel label
    ax.text(0.5, 1.05, label, transform=ax.transAxes,
            fontsize=60, fontweight='bold', va='bottom', ha='center')

    for bar1, bar2, p, or_val in zip(bars1, bars2, data['P_Value'], data['Odds_Ratio']):
        draw_significance_bracket(ax, bar1, bar2, p, or_val, fontsize=26)

# Legend stays at fontsize=22, positioned safely below everything
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=22, ncol=2,
           bbox_to_anchor=(0.5, -0.05), frameon=True)

# Reserve more room at the bottom so the legend doesn't overlap x-tick labels
fig.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig(GRAPH_OUTPUT_DIR / 'supplementary_figure_3_combined_figure.jpeg',
            dpi=600, bbox_inches='tight')
print('Saved: supplementary_figure_3_combined_figure.jpeg')


# ## Summary
# 
# All figures saved to `results/3_interaction_overlap_graphs/`.

# 