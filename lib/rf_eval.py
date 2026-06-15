"""Shared library — evaluation of a validation screen (AUROC/AUPR + ROC/PR plots).

Two modes:
  * Sourced (`score_col` set): the validation table already carries the model
    probability sourced from the matching clinical-trials prediction CSV — no RF
    is trained, the pre-computed scores are evaluated directly. Identical to a
    local retrain because clinical and validation share TRAIN_CSV, RF
    hyperparameters, feature order and seed.
  * Legacy (`score_col=None`): trains a RandomForest on `train_csv`, scores
    `val_features_csv`, and auto-detects an SL_Pair (`excl_sl_pair`) for a second
    leakage-excluded eval.

Both write ROC + PR plots (Arial, no title, sorted legend, STRING/Random/Baseline
references) and AUROC/AUPR with bootstrap CIs.

Public entry point:
  run_screen_eval(train_csv, val_features_csv, labels_csv, screen_name,
                   out_plot, out_json, feature_cols, target_col, rf_kw,
                   excl_sl_pair=None, out_plot_excl=None, known_hits=None,
                   skip_full=False, rf_color=None, score_col=None) -> dict
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

COLOR_STRING = "#CC79A7"


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=42):
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    pt   = float(metric_fn(y_true, y_score))
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            vals[i] = np.nan
            continue
        vals[i] = metric_fn(yt, y_score[idx])
    return pt, float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def relabel(val_base: pd.DataFrame, labels_csv: Path, screen_name: str,
            target_col: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    lbl = (df.groupby("query_gene")["label"]
             .apply(lambda s: "Resistance" if (s == "Resistance").any() else "Non-Resistance"))
    val = val_base.copy()
    val["Class"] = val["Query"].map(lbl).fillna("Non-Resistance")
    val[target_col] = (val["Class"] == "Resistance").astype(int)
    val["Screen"] = screen_name
    return val


def eval_one(val: pd.DataFrame, probs, label: str,
             out_path: Path, feature_cols: list, target_col: str,
             screen_display_name: str, rf_color: str = None) -> dict:
    y     = val[target_col].values
    probs = np.asarray(probs, dtype=float)
    str_s = val["StringInteractionWithBiomarker"].fillna(0).values

    auroc, lo, hi   = bootstrap_ci(y, probs, roc_auc_score)
    aupr,  plo, phi = bootstrap_ci(y, probs, average_precision_score)
    str_auroc = float(roc_auc_score(y, str_s))
    str_aupr  = float(average_precision_score(y, str_s))
    base_ap   = float(y.mean())

    print(f"  {label}: n={len(y)} n_pos={int(y.sum())} "
          f"AUROC={auroc:.4f} [{lo:.4f},{hi:.4f}]  "
          f"AUPR={aupr:.4f}  STRING_AUROC={str_auroc:.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    # --- ROC ---
    ax = axs[0]
    fpr, tpr, _ = roc_curve(y, probs)
    sfpr, stpr, _ = roc_curve(y, str_s)
    handles, labels_, aucs = [], [], []

    rf_kwargs = {"color": rf_color} if rf_color else {}
    line, = ax.plot(fpr, tpr, linewidth=2, label=f"{screen_display_name} (AUC = {auroc:.2f})", **rf_kwargs)
    handles.append(line); labels_.append(line.get_label()); aucs.append(auroc)

    line, = ax.plot(sfpr, stpr, linestyle="--", linewidth=2, color=COLOR_STRING,
                    label=f"STRING (Q-B (AUC = {str_auroc:.2f}))")
    handles.append(line); labels_.append(line.get_label()); aucs.append(str_auroc)

    line_rand, = ax.plot([0, 1], [0, 1], color="gray", linestyle="-.", linewidth=1.5,
                         label="Random (AUC = 0.50)")

    sorted_items = sorted(zip(handles, labels_, aucs), key=lambda x: x[2], reverse=True)
    sh, sl, _ = zip(*sorted_items)
    ax.legend(list(sh) + [line_rand], list(sl) + [line_rand.get_label()],
              loc="lower right", fontsize=15, frameon=True,
              edgecolor="black", fancybox=False)

    ax.set_xlabel("False Positive Rate", fontsize=19)
    ax.set_ylabel("True Positive Rate", fontsize=19)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- PR ---
    ax = axs[1]
    prec, rec, _ = precision_recall_curve(y, probs)
    sprec, srec, _ = precision_recall_curve(y, str_s)
    handles, labels_, aps = [], [], []

    line, = ax.plot(rec, prec, linewidth=2, label=f"{screen_display_name} (AP = {aupr:.2f})", **rf_kwargs)
    handles.append(line); labels_.append(line.get_label()); aps.append(aupr)

    line, = ax.plot(srec, sprec, linestyle="--", linewidth=2, color=COLOR_STRING,
                    label=f"STRING (Q-B (AP = {str_aupr:.2f}))")
    handles.append(line); labels_.append(line.get_label()); aps.append(str_aupr)

    line_base = ax.hlines(base_ap, 0, 1, linestyle="-.", color="gray",
                          label=f"Baseline (AP = {base_ap:.2f})", linewidth=1.2)

    sorted_items = sorted(zip(handles, labels_, aps), key=lambda x: x[2], reverse=True)
    sh, sl, _ = zip(*sorted_items)
    ax.legend(list(sh) + [line_base], list(sl) + [line_base.get_label()],
              loc="upper right", fontsize=15, frameon=True,
              edgecolor="black", fancybox=False)

    ax.set_xlabel("Recall", fontsize=19)
    ax.set_ylabel("Precision", fontsize=19)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}.png/.pdf")

    _SEP = (10, 7)
    _ADJ = dict(left=0.13, right=0.97, top=0.95, bottom=0.13)

    # --- Separate ROC ---
    roc_stem = out_path.parent / (out_path.stem + "_roc")
    fig_r, ax_r = plt.subplots(figsize=_SEP)
    line, = ax_r.plot(fpr, tpr, linewidth=2,
                      label=f"{screen_display_name} (AUC = {auroc:.2f})", **rf_kwargs)
    handles_r = [line]; labels_r = [line.get_label()]; aucs_r = [auroc]
    line, = ax_r.plot(sfpr, stpr, linestyle="--", linewidth=2, color=COLOR_STRING,
                      label=f"STRING (Q-B (AUC = {str_auroc:.2f}))")
    handles_r.append(line); labels_r.append(line.get_label()); aucs_r.append(str_auroc)
    line_rand_r, = ax_r.plot([0, 1], [0, 1], color="gray", linestyle="-.", linewidth=1.5,
                              label="Random (AUC = 0.50)")
    si = sorted(zip(handles_r, labels_r, aucs_r), key=lambda x: x[2], reverse=True)
    sh, sl, _ = zip(*si)
    ax_r.legend(list(sh) + [line_rand_r], list(sl) + [line_rand_r.get_label()],
                loc="lower right", fontsize=15, frameon=True, edgecolor="black", fancybox=False)
    ax_r.set_xlabel("False Positive Rate", fontsize=19)
    ax_r.set_ylabel("True Positive Rate", fontsize=19)
    ax_r.tick_params(axis="both", labelsize=15)
    ax_r.grid(False)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    fig_r.subplots_adjust(**_ADJ)
    fig_r.savefig(roc_stem.with_suffix(".pdf"))
    fig_r.savefig(roc_stem.with_suffix(".png"), dpi=600)
    plt.close(fig_r)
    print(f"  wrote {roc_stem}.png/.pdf")

    # --- Separate PR ---
    pr_stem = out_path.parent / (out_path.stem + "_pr")
    fig_p, ax_p = plt.subplots(figsize=_SEP)
    line, = ax_p.plot(rec, prec, linewidth=2,
                      label=f"{screen_display_name} (AP = {aupr:.2f})", **rf_kwargs)
    handles_p = [line]; labels_p = [line.get_label()]; aps_p = [aupr]
    line, = ax_p.plot(srec, sprec, linestyle="--", linewidth=2, color=COLOR_STRING,
                      label=f"STRING (Q-B (AP = {str_aupr:.2f}))")
    handles_p.append(line); labels_p.append(line.get_label()); aps_p.append(str_aupr)
    line_base_p = ax_p.hlines(base_ap, 0, 1, linestyle="-.", color="gray",
                               label=f"Baseline (AP = {base_ap:.2f})", linewidth=1.2)
    si = sorted(zip(handles_p, labels_p, aps_p), key=lambda x: x[2], reverse=True)
    sh, sl, _ = zip(*si)
    ax_p.legend(list(sh) + [line_base_p], list(sl) + [line_base_p.get_label()],
                loc="upper right", fontsize=15, frameon=True, edgecolor="black", fancybox=False)
    ax_p.set_xlabel("Recall", fontsize=19)
    ax_p.set_ylabel("Precision", fontsize=19)
    ax_p.tick_params(axis="both", labelsize=15)
    ax_p.grid(False)
    ax_p.spines["top"].set_visible(False)
    ax_p.spines["right"].set_visible(False)
    fig_p.subplots_adjust(**_ADJ)
    fig_p.savefig(pr_stem.with_suffix(".pdf"))
    fig_p.savefig(pr_stem.with_suffix(".png"), dpi=600)
    plt.close(fig_p)
    print(f"  wrote {pr_stem}.png/.pdf")

    return dict(
        variant=label,
        n=int(len(y)), n_pos=int(y.sum()), baseline_ap=base_ap,
        auroc=auroc, auroc_ci=[lo, hi],
        aupr=aupr,   aupr_ci=[plo, phi],
        string_auroc=str_auroc, string_aupr=str_aupr,
    )


def run_screen_eval(
    train_csv: Path,
    val_features_csv: Path,
    labels_csv: Path,
    screen_name: str,
    screen_display_name: str,
    variant_label: str,
    out_plot: Path,
    out_json: Path,
    feature_cols: list,
    target_col: str,
    rf_kw: dict,
    excl_sl_pair: str = None,
    out_plot_excl: Path = None,
    known_hits: list = None,
    skip_full: bool = False,
    rf_color: str = None,
    score_col: str = None,
) -> dict:
    """Eval on `val_features_csv` relabeled via `labels_csv`.

    If `score_col` is given, the validation table already carries the model's
    probability (sourced from the matching clinical-trials prediction CSV); no RF
    is trained — the sourced scores are used directly (clinical and validation
    share the same TRAIN_CSV, RF hyperparameters, feature order and seed, so the
    scores are identical to a local retrain). Otherwise the legacy training path
    runs: train RF on `train_csv`, auto-detect `excl_sl_pair`, optional `skip_full`.
    """
    out_plot = Path(out_plot)
    out_json = Path(out_json)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    val_base = pd.read_csv(val_features_csv, low_memory=False)
    print(f"  val_base: {val_base.shape}")

    val = relabel(val_base, labels_csv, screen_name, target_col)
    n_pos = int(val[target_col].sum())
    n_neg = int((val[target_col] == 0).sum())
    print(f"  val labeled: {val.shape}  pos={n_pos}  neg={n_neg}")

    out_dict = {}
    val_known = val[val["Query"].isin(known_hits or [])].copy()

    # --- Sourced path: clinical predictions used directly, no RF training ---
    if score_col is not None:
        if score_col not in val.columns:
            raise ValueError(
                f"score_col '{score_col}' not in validation features — re-run "
                f"run_features so it carries the clinical prediction column."
            )
        probs = val[score_col].astype(float).values
        print(f"\n=== {screen_display_name} {variant_label} (clinical-sourced scores) ===")
        r = eval_one(val, probs, variant_label, out_plot, feature_cols, target_col,
                     screen_display_name, rf_color=rf_color)
        if len(val_known):
            vk = val_known.copy()
            vk["RF_score"] = vk[score_col].astype(float).values
            print(f"\n=== Known-hit scores (clinical-sourced) ===")
            print(vk[["Query", "RF_score", target_col]]
                  .sort_values("RF_score", ascending=False).to_string(index=False))
        out_dict[f"{variant_label}_clinical_sourced"] = r
        out_json.write_text(json.dumps(out_dict, indent=2))
        print(f"\n  wrote {out_json}")
        return out_dict

    # --- Legacy training path (score_col=None) ---
    train = pd.read_csv(train_csv, low_memory=False)
    print(f"  train: {train.shape}  pos={int(train[target_col].sum())}")

    if not skip_full:
        rf = RandomForestClassifier(**rf_kw)
        rf.fit(train[feature_cols].values, train[target_col].values)
        print("  RF trained (full)")

        print(f"\n=== {screen_display_name} {variant_label} (full training) ===")
        probs_full = rf.predict_proba(val[feature_cols].values)[:, 1]
        r_full = eval_one(val, probs_full, variant_label, out_plot, feature_cols, target_col,
                          screen_display_name, rf_color=rf_color)

        if len(val_known):
            val_known["RF_score"] = rf.predict_proba(val_known[feature_cols].values)[:, 1]
            print(f"\n=== Known-hit RF scores (full training) ===")
            print(val_known[["Query", "RF_score", target_col]]
                  .sort_values("RF_score", ascending=False)
                  .to_string(index=False))

        out_dict[f"{variant_label}_full"] = r_full

    if excl_sl_pair:
        excl_present = (train["SL_Pair"] == excl_sl_pair).sum()
        if excl_present > 0 and out_plot_excl is not None:
            train_excl = train[train["SL_Pair"] != excl_sl_pair].reset_index(drop=True)
            print(f"\n--- Excluding {excl_sl_pair}: {len(train)} → {len(train_excl)} rows "
                  f"(removed {excl_present}) ---")

            rf_excl = RandomForestClassifier(**rf_kw)
            rf_excl.fit(train_excl[feature_cols].values, train_excl[target_col].values)
            print("  RF (excl) trained")

            print(f"\n=== {screen_display_name} {variant_label} (excl {excl_sl_pair}) ===")
            probs_excl = rf_excl.predict_proba(val[feature_cols].values)[:, 1]
            r_excl = eval_one(val, probs_excl, variant_label, Path(out_plot_excl),
                              feature_cols, target_col, screen_display_name, rf_color=rf_color)

            if len(val_known):
                val_known["RF_score_excl"] = rf_excl.predict_proba(val_known[feature_cols].values)[:, 1]
                print(f"\n=== Known-hit RF scores (excl {excl_sl_pair}) ===")
                cols = ["Query"]
                if "RF_score" in val_known.columns:
                    cols.append("RF_score")
                cols.extend(["RF_score_excl", target_col])
                print(val_known[cols]
                      .sort_values("RF_score_excl", ascending=False)
                      .to_string(index=False))

            out_dict[f"{variant_label}_excl_{excl_sl_pair}"] = r_excl
        else:
            print(f"\n  {excl_sl_pair} not present in training — skipping exclusion variant")

    out_json.write_text(json.dumps(out_dict, indent=2))
    print(f"\n  wrote {out_json}")
    return out_dict
