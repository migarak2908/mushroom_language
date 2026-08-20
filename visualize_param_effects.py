"""
How each swept environment parameter relates to the emergence of
discrimination, feedforward (no pain/pleasure) vs. recurrent (+pain/pleasure)
overlaid -- the two full 500-run sweeps, not the 20-config ablation.

Reads the per-config summary CSVs each sweep already produces (sweep_plot.py's
config_summary_probed.csv, or the equivalent table you pasted earlier as
feedforward.tsv / recurrent.tsv) and, for each of the 4 swept parameters x
each of 2 outcome metrics (population-level env_disc, individual-level probe
eat_disc), plots every config as a point plus a quantile-binned trend line
per condition. A plain scatter of 100 LHS points per condition is too sparse
to eyeball a relationship in, and any single functional-form fit (linear,
etc.) would be presuming a shape that isn't given -- binning by quantile
and connecting bin means is a assumption-light way to see the shape of the
relationship instead.

Also prints (and saves) a Spearman correlation table across all
param x metric x condition combinations, since a number is easier to put a
sentence around than "the trend line curves upward a bit."
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
FF_SUMMARY_PATH = "/content/drive/MyDrive/sweep/config_summary_probed.csv"
REC_SUMMARY_PATH = "/content/drive/MyDrive/sweep_recurrent/config_summary_probed.csv"
OUT_DIR = "."

PARAMS = ["mutation_std", "poison_multiplier", "reprod_cost", "energy_decay"]
LOG_SCALE_PARAMS = {"mutation_std"}  # sampled log-uniform in the LHS design

METRICS = [
    ("mean_final_env_disc", "population-level (env)"),
    ("mean_final_probe_eat_disc", "individual-level (probe eat)"),
]

N_BINS = 8

FF_COLOR = "#eb6834"
REC_COLOR = "#2a78d6"
GRID = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"


def binned_trend(x, y, n_bins=N_BINS):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < n_bins * 2:
        n_bins = max(2, len(df) // 3)
    df["bin"] = pd.qcut(df["x"], n_bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean")).sort_values("x")
    return grouped["x"].values, grouped["y"].values


def plot_panel(ax, ff, rec, param, metric_col):
    ax.scatter(ff[param], ff[metric_col], s=16, color=FF_COLOR, alpha=0.45, edgecolors="none", label="feedforward")
    ax.scatter(rec[param], rec[metric_col], s=16, color=REC_COLOR, alpha=0.45, edgecolors="none", label="recurrent+pp")

    bx, by = binned_trend(ff[param].values, ff[metric_col].values)
    ax.plot(bx, by, color=FF_COLOR, linewidth=2.2, marker="o", markersize=4)

    bx, by = binned_trend(rec[param].values, rec[metric_col].values)
    ax.plot(bx, by, color=REC_COLOR, linewidth=2.2, marker="o", markersize=4)

    ax.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    if param in LOG_SCALE_PARAMS:
        ax.set_xscale("log")
    ax.set_facecolor("#fcfcfb")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def correlation_table(ff, rec):
    rows = []
    for metric_col, metric_label in METRICS:
        for param in PARAMS:
            for cond_name, df in [("feedforward", ff), ("recurrent+pp", rec)]:
                valid = df[[param, metric_col]].dropna()
                r, p = stats.spearmanr(valid[param], valid[metric_col])
                rows.append({
                    "metric": metric_label, "param": param, "condition": cond_name,
                    "spearman_r": r, "p_value": p, "n": len(valid),
                })
    return pd.DataFrame(rows)


def main():
    ff = pd.read_csv(FF_SUMMARY_PATH)
    rec = pd.read_csv(REC_SUMMARY_PATH)

    fig, axes = plt.subplots(len(METRICS), len(PARAMS), figsize=(4.2 * len(PARAMS), 4.0 * len(METRICS)))
    fig.patch.set_facecolor("#fcfcfb")

    for row, (metric_col, metric_label) in enumerate(METRICS):
        for col, param in enumerate(PARAMS):
            ax = axes[row, col]
            plot_panel(ax, ff, rec, param, metric_col)
            if row == len(METRICS) - 1:
                ax.set_xlabel(param, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{metric_label}\ndiscrimination", fontsize=9)
            if row == 0:
                ax.set_title(param, fontsize=10, color=INK)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="-", color=FF_COLOR, markersize=5, label="feedforward (no pain/pleasure)"),
        plt.Line2D([], [], marker="o", linestyle="-", color=REC_COLOR, markersize=5, label="recurrent (+pain/pleasure)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(OUT_DIR, "param_effects_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    corr = correlation_table(ff, rec)
    corr_path = os.path.join(OUT_DIR, "param_effects_correlations.csv")
    corr.to_csv(corr_path, index=False)
    print(f"\nSaved {corr_path}\n")
    print(corr.pivot_table(index=["metric", "param"], columns="condition", values="spearman_r").round(3).to_string())


if __name__ == "__main__":
    main()
