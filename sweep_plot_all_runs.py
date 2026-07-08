"""Plot probe_approach_disc and probe_eat_disc across every run in the sweep at once,
rather than broken out per-config (see sweep_plot.py for the per-config view).

For each metric, produces:
  - a spaghetti plot: every run's full time series as a thin line, colored by whether
    that run went extinct, with a bold mean-across-runs trend line on top
  - a distribution plot: one point per run (its final windowed value), as a jittered
    strip plot, so you can see the overall spread of outcomes across all 500 runs
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

NO_SIGNAL = True  # must match the flag used when sweep.py was run
RESULTS_DIR = f"/content/drive/MyDrive/sweep{'_nosig' if NO_SIGNAL else ''}"

BLUE = "#2a78d6"       # approach
RED = "#e34948"        # eat
CRITICAL = "#d03b3b"   # extinct runs
GRID = "#c3c2b7"

METRICS = [
    ("probe_approach_disc", BLUE, "approach disc"),
    ("probe_eat_disc", RED, "eat disc"),
]


def final_stat(values, frac=0.2):
    """Mean over the last `frac` of a series -- same windowing as sweep_plot.py's
    final_stat, so this is comparable to that script's per-config summaries."""
    vals = [v for v in values[int(len(values) * (1 - frac)):] if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


# Load every run flat (not grouped by config -- we want all 500 at once here)
runs = []
for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.startswith("cfg") or not fname.endswith(".json") or fname == "configs.json":
        continue
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        data = json.load(f)
    runs.append(data)

print(f"Loaded {len(runs)} runs from {RESULTS_DIR}")

for metric_key, color, label in METRICS:
    extinct_runs = [r for r in runs if r["extinction_chunk"] is not None]
    survived_runs = [r for r in runs if r["extinction_chunk"] is None]

    # --- Spaghetti plot: every run's full time series ---
    fig, ax = plt.subplots(figsize=(9, 5))

    # bin trend line by exact probe chunk value, so extinct runs (fewer points)
    # simply drop out of the average past their extinction point
    trend = {}
    for r in runs:
        for c, v in zip(r["probe_chunks"], r[metric_key]):
            if not np.isnan(v):
                trend.setdefault(c, []).append(v)

    for r in survived_runs:
        ax.plot(r["probe_chunks"], r[metric_key], color=color, alpha=0.08, linewidth=0.8)
    for r in extinct_runs:
        ax.plot(r["probe_chunks"], r[metric_key], color=CRITICAL, alpha=0.08, linewidth=0.8)

    trend_chunks = sorted(trend)
    trend_vals = [np.mean(trend[c]) for c in trend_chunks]
    ax.plot(trend_chunks, trend_vals, color=color, linewidth=2.5, label="mean across runs", zorder=5)

    ax.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax.set_xlabel("chunk (x100 env steps)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} across all {len(runs)} runs "
                 f"({len(survived_runs)} survived, {len(extinct_runs)} extinct)", fontsize=10)
    ax.plot([], [], color=color, alpha=0.5, linewidth=1.5, label="survived run")
    ax.plot([], [], color=CRITICAL, alpha=0.5, linewidth=1.5, label="extinct run")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"all_runs_{metric_key}_timeseries.png"), dpi=130)
    plt.close(fig)

    # --- Distribution: one final windowed value per run ---
    final_survived = [final_stat(r[metric_key]) for r in survived_runs]
    final_extinct = [final_stat(r[metric_key]) for r in extinct_runs]
    final_survived = [v for v in final_survived if not np.isnan(v)]
    final_extinct = [v for v in final_extinct if not np.isnan(v)]

    fig, ax = plt.subplots(figsize=(4.5, 5.5))
    rng = np.random.default_rng(0)

    if final_survived:
        jitter = rng.uniform(-0.15, 0.15, size=len(final_survived))
        ax.scatter(np.zeros(len(final_survived)) + jitter, final_survived,
                   color=color, alpha=0.5, s=18, edgecolors="none")
        ax.plot([-0.2, 0.2], [np.median(final_survived)] * 2, color="#0b0b0b", linewidth=2)

    if final_extinct:
        jitter = rng.uniform(-0.15, 0.15, size=len(final_extinct))
        ax.scatter(np.ones(len(final_extinct)) + jitter, final_extinct,
                   color=CRITICAL, alpha=0.5, s=18, edgecolors="none")
        ax.plot([0.8, 1.2], [np.median(final_extinct)] * 2, color="#0b0b0b", linewidth=2)

    ax.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"survived\n(n={len(final_survived)})", f"extinct\n(n={len(final_extinct)})"])
    ax.set_ylabel(f"final {label} (mean of last 20% of probe series)")
    ax.set_title(f"final {label} per run", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"all_runs_{metric_key}_distribution.png"), dpi=130)
    plt.close(fig)

    print(f"{label}: survived mean={np.mean(final_survived):.4f} (n={len(final_survived)}), "
          f"extinct mean={np.mean(final_extinct) if final_extinct else float('nan'):.4f} (n={len(final_extinct)})")

print("Done.")
