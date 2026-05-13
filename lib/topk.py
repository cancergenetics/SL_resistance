"""Shared library — top-k prioritization analysis (within-biomarker Monte Carlo).

Simulates clinical scenario: patient with biomarker-deficient background develops
resistance; N mutated genes found in tumor; can the model rank the causal
resistance gene into the top-k?

Public entry point:
  run_screen_topk(train_csv, val_features_csv, labels_csv, screen_name,
                   out_stem, out_json, feature_cols, target_col, rf_kw,
                   n_list, k_list, n_trials, seed, plot_title)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DEFAULT_N_LIST   = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
DEFAULT_K_LIST   = [1, 3]
DEFAULT_N_TRIALS = 20_000
DEFAULT_SEED     = 42
DEFAULT_COLORS   = {1: "#ff7f0e", 3: "#d62728"}   # orange / red (Knoll palette)


def relabel(val_base: pd.DataFrame, labels_csv: Path, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    lbl = (df.groupby("query_gene")["label"]
             .apply(lambda s: "Resistance" if (s == "Resistance").any() else "Non-Resistance"))
    val = val_base.copy()
    val["Class"] = val["Query"].map(lbl).fillna("Non-Resistance")
    val[target_col] = (val["Class"] == "Resistance").astype(int)
    return val


def empirical_hitk_within_biomarker(
    val: pd.DataFrame,
    y_prob: np.ndarray,
    target_col: str,
    n_list: list,
    k_list: list,
    trials: int,
    seed: int,
) -> tuple:
    import sys
    import time

    df = val[["Biomarker", target_col]].copy()
    df["score"] = y_prob
    rng  = np.random.default_rng(seed)
    pos  = df[df[target_col] == 1].reset_index(drop=True)
    global_neg_scores = df[df[target_col] == 0]["score"].values

    results = {}
    errors = {}

    total_combos = len(n_list) * len(k_list)
    combo_idx = 0
    t_start = time.time()
    progress_step = max(1, trials // 10)   # print 10× per (N,k)

    for N in n_list:
        for k in k_list:
            combo_idx += 1
            t_combo = time.time()
            print(f"  [{combo_idx}/{total_combos}] N={N} k={k} starting "
                  f"({trials} trials) …", flush=True)

            hits = np.empty(trials, dtype=bool)
            for i in range(trials):
                if i and i % progress_step == 0:
                    pct = 100 * i / trials
                    print(f"    progress: {i}/{trials} ({pct:.0f}%)", flush=True)

                idx   = rng.integers(len(pos))
                s_pos = pos.iloc[idx]["score"]

                replace = len(global_neg_scores) < N - 1
                s_neg = rng.choice(global_neg_scores, size=N - 1, replace=replace)
                rank  = 1 + int(np.sum(s_neg > s_pos))
                hits[i] = rank <= k

            mean_hit = float(hits.mean())
            results[(N, k)] = mean_hit
            errors[(N, k)]  = float(hits.std(ddof=1) / np.sqrt(trials))
            elapsed = time.time() - t_combo
            print(f"    done in {elapsed:.1f}s — top-{k}@N={N} = {mean_hit:.4f}",
                  flush=True)

    total_elapsed = time.time() - t_start
    print(f"\n  Total Monte Carlo time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)",
          flush=True)
    return results, errors


def plot_topk(res: dict, err: dict, title, out_stem: Path,
              n_list: list, k_list: list, colors: dict = None) -> None:
    colors = colors or DEFAULT_COLORS
    fig, ax = plt.subplots(figsize=(10, 7))
    for k in k_list:
        y      = [res[(N, k)] for N in n_list]
        yerr   = [err[(N, k)] for N in n_list]
        y_rand = [k / N for N in n_list]
        ax.errorbar(n_list, y, yerr=yerr, marker="o", lw=2, capsize=6,
                    capthick=2, elinewidth=1.5, color=colors[k], label=f"Top-{k}")
        ax.plot(n_list, y_rand, lw=2, linestyle="--", color=colors[k],
                alpha=0.5, label=f"Random (Top-{k})")
    ax.set_xlabel("Number of mutated genes (N)", fontsize=19)
    ax.set_ylabel("P(causal resistance gene in top-k)", fontsize=19)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=15, frameon=True, edgecolor="black", fancybox=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=16)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.13)
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
    print(f"  wrote {out_stem}.png/.pdf")


def run_screen_topk(
    train_csv: Path,
    val_features_csv: Path,
    labels_csv: Path,
    screen_name: str,
    out_stem: Path,
    out_json: Path,
    feature_cols: list,
    target_col: str,
    rf_kw: dict,
    plot_title: str = None,
    n_list: list = None,
    k_list: list = None,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = DEFAULT_SEED,
    excl_sl_pair: str = None,
    colors: dict = None,
) -> dict:
    n_list = n_list or DEFAULT_N_LIST
    k_list = k_list or DEFAULT_K_LIST
    out_stem = Path(out_stem)
    out_json = Path(out_json)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    train = pd.read_csv(train_csv, low_memory=False)
    if excl_sl_pair:
        n_before = len(train)
        train = train[train["SL_Pair"] != excl_sl_pair].reset_index(drop=True)
        print(f"  excluded {excl_sl_pair}: {n_before} → {len(train)} rows")
    val_base = pd.read_csv(val_features_csv, low_memory=False)

    val = relabel(val_base, labels_csv, target_col)
    n_pos = int(val[target_col].sum())
    n_neg = int((val[target_col] == 0).sum())
    print(f"  train: {train.shape}  val: {val.shape}  pos={n_pos}  neg={n_neg}")

    rf = RandomForestClassifier(**rf_kw)
    rf.fit(train[feature_cols].values, train[target_col].values)
    y_prob = rf.predict_proba(val[feature_cols].values)[:, 1]
    print("  RF scored")

    print(f"  Running top-k simulation (N_TRIALS={n_trials}) …")
    res, err = empirical_hitk_within_biomarker(val, y_prob, target_col,
                                                n_list, k_list, n_trials, seed)

    print("\n  Results:")
    print(f"  {'N':>4}  {'top-1':>8}  {'top-3':>8}  {'rand-1':>8}  {'rand-3':>8}")
    for N in n_list:
        print(f"  {N:>4}  {res[(N,1)]:>8.4f}  {res[(N,3)]:>8.4f}  {1/N:>8.4f}  {3/N:>8.4f}")

    out = dict(
        name=screen_name,
        n_positives=n_pos, n_negatives=n_neg,
        n_trials=n_trials, N_list=n_list,
        top1={str(N): res[(N, 1)] for N in n_list},
        top3={str(N): res[(N, 3)] for N in n_list},
        top1_err={str(N): err[(N, 1)] for N in n_list},
        top3_err={str(N): err[(N, 3)] for N in n_list},
    )
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {out_json}")

    plot_topk(res, err, plot_title, out_stem, n_list, k_list, colors=colors)
    return out
