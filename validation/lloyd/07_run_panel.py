"""Lloyd validation — panel: ROC (C) | top-k (D).

Trains RF (excl ARID1A_ATR).
Output: results/lloyd_panel.png / .pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    EXCL_SL_PAIR,
    FEATURE_COLS,
    FEATURES_DROPNA_DEDUP_CSV,
    LABELS_IC90_SUM_HUMAN_CSV,
    RESULTS,
    RF_KW,
    TARGET_COL,
    TRAIN_CSV,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RF_COLOR     = "#2ca02c"
COLOR_STRING = "#CC79A7"
TOPK_COLORS  = {1: "#ff7f0e", 3: "#d62728"}
PANEL_W, PANEL_H = 10, 7
LABEL_FS = 40
AXIS_FS  = 19
TICK_FS  = 15
LEG_FS   = 15


def relabel(val_base, labels_csv, target_col):
    df  = pd.read_csv(labels_csv)
    lbl = (df.groupby("query_gene")["label"]
             .apply(lambda s: "Resistance" if (s == "Resistance").any() else "Non-Resistance"))
    val = val_base.copy()
    val["Class"]    = val["Query"].map(lbl).fillna("Non-Resistance")
    val[target_col] = (val["Class"] == "Resistance").astype(int)
    return val


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train = pd.read_csv(TRAIN_CSV, low_memory=False)
    train = train[train["SL_Pair"] != EXCL_SL_PAIR].reset_index(drop=True)
    print(f"  train after excl {EXCL_SL_PAIR}: {train.shape}")

    val_base = pd.read_csv(FEATURES_DROPNA_DEDUP_CSV, low_memory=False)
    val      = relabel(val_base, LABELS_IC90_SUM_HUMAN_CSV, TARGET_COL)
    y        = val[TARGET_COL].values
    print(f"  val: {val.shape}  pos={int(y.sum())}  neg={int((y==0).sum())}")

    rf = RandomForestClassifier(**RF_KW)
    rf.fit(train[FEATURE_COLS].values, train[TARGET_COL].values)
    print("  RF trained")

    probs = rf.predict_proba(val[FEATURE_COLS].values)[:, 1]
    str_s = val["StringInteractionWithBiomarker"].fillna(0).values

    auroc     = float(roc_auc_score(y, probs))
    str_auroc = float(roc_auc_score(y, str_s))
    fpr,  tpr,  _ = roc_curve(y, probs)
    sfpr, stpr, _ = roc_curve(y, str_s)

    with open(RESULTS / "topk_lloyd.json") as f:
        tk = json.load(f)
    N_list = tk["N_list"]

    fig, axs = plt.subplots(1, 2, figsize=(PANEL_W * 2, PANEL_H))

    # --- C: ROC ---
    ax = axs[0]
    ax.text(-0.1, 1.05, "B", transform=ax.transAxes,
            fontsize=LABEL_FS, fontweight="bold", va="top")
    handles, labels_, aucs = [], [], []
    line, = ax.plot(fpr, tpr, linewidth=2, color=RF_COLOR,
                    label=f"Lloyd (AUC = {auroc:.2f})")
    handles.append(line); labels_.append(line.get_label()); aucs.append(auroc)
    line, = ax.plot(sfpr, stpr, linestyle="--", linewidth=2, color=COLOR_STRING,
                    label=f"STRING (Q-B (AUC = {str_auroc:.2f}))")
    handles.append(line); labels_.append(line.get_label()); aucs.append(str_auroc)
    line_rand, = ax.plot([0, 1], [0, 1], color="gray", linestyle="-.", linewidth=1.5,
                         label="Random (AUC = 0.50)")
    si = sorted(zip(handles, labels_, aucs), key=lambda x: x[2], reverse=True)
    sh, sl, _ = zip(*si)
    ax.legend(list(sh) + [line_rand], list(sl) + [line_rand.get_label()],
              loc="lower right", fontsize=LEG_FS, frameon=True, edgecolor="black", fancybox=False)
    ax.set_xlabel("False Positive Rate", fontsize=AXIS_FS)
    ax.set_ylabel("True Positive Rate", fontsize=AXIS_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- D: top-k ---
    ax = axs[1]
    ax.text(-0.1, 1.05, "D", transform=ax.transAxes,
            fontsize=LABEL_FS, fontweight="bold", va="top")
    for k, key, err_key in [(1, "top1", "top1_err"), (3, "top3", "top3_err")]:
        color  = TOPK_COLORS[k]
        y_vals = [tk[key][str(N)] for N in N_list]
        y_err  = [tk[err_key][str(N)] for N in N_list]
        y_rand = [k / N for N in N_list]
        ax.errorbar(N_list, y_vals, yerr=y_err, marker="o", lw=2, capsize=6,
                    capthick=2, elinewidth=1.5, color=color, label=f"Top-{k}")
        ax.plot(N_list, y_rand, lw=2, linestyle="--", color=color,
                alpha=0.5, label=f"Random (Top-{k})")
    ax.set_xlabel("Number of mutated genes (N)", fontsize=AXIS_FS)
    ax.set_ylabel("P(causal resistance gene in top-k)", fontsize=AXIS_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.legend(fontsize=LEG_FS, frameon=True, edgecolor="black", fancybox=False)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = RESULTS / "lloyd_panel"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}.png/.pdf")


if __name__ == "__main__":
    main()
