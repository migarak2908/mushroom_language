"""Plot probe_approach_disc and probe_eat_disc across every run in the sweep at once,
rather than broken out per-config (see sweep_plot.py for the per-config view).

Runs belonging to a config whose extinction rate exceeds MAX_CONFIG_EXTINCTION_RATE
are dropped entirely before any of the plots below are made.

For each metric, produces:
  - a spaghetti plot: every remaining run's full time series as a thin line, colored
    by whether that run went extinct, with a bold mean-across-runs trend line on top
  - a distribution plot: one point per run (its final windowed value), as a jittered
    strip plot, so you can see the overall spread of outcomes
  - a top-N plot: the TOP_N runs by final windowed value, each an individually
    identifiable line (labeled by config/seed)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json

import numpy as np
import matplotlib.pyplot as plt

NO_SIGNAL = True  # must match the flag used when sweep.py was run
RESULTS_DIR = f"/content/drive/MyDrive/sweep{'_nosig' if NO_SIGNAL else ''}"

TOP_N = 10
MAX_CONFIG_EXTINCTION_RATE = 0.4  # drop whole configs where more than this fraction of seeds went extinct

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

# Drop runs belonging to configs whose extinction rate is too high -- a config that
# mostly dies isn't one you'd actually use, even if one lucky seed did well
by_config = {}
for r in runs:
    by_config.setdefault(r["config"]["config_id"], []).append(r)

stable_config_ids = {
    cfg_id for cfg_id, rs in by_config.items()
    if np.mean([r["extinction_chunk"] is not None for r in rs]) <= MAX_CONFIG_EXTINCTION_RATE
}
n_before = len(runs)
runs = [r for r in runs if r["config"]["config_id"] in stable_config_ids]
print(f"Dropped {n_before - len(runs)} runs from "
      f"{len(by_config) - len(stable_config_ids)} configs with extinction rate > {MAX_CONFIG_EXTINCTION_RATE}; "
      f"{len(runs)} runs remain across {len(stable_config_ids)} configs.")

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

    # --- Top N runs by final value: full trajectories, individually identifiable ---
    ranked = sorted(
        ((final_stat(r[metric_key]), r) for r in runs),
        key=lambda t: t[0], reverse=True,
    )
    ranked = [(v, r) for v, r in ranked if not np.isnan(v)]
    top_n = ranked[:TOP_N]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, (v, r) in enumerate(top_n):
        cfg_id = r["config"]["config_id"]
        seed = r["seed"]
        extinct_tag = " [extinct]" if r["extinction_chunk"] is not None else ""
        ax.plot(r["probe_chunks"], r[metric_key], color=cmap(i % 10), linewidth=1.6,
                label=f"cfg{cfg_id:03d} seed{seed} ({v:.3f}){extinct_tag}")

    ax.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax.set_xlabel("chunk (x100 env steps)")
    ax.set_ylabel(label)
    ax.set_title(f"top {TOP_N} runs by final {label} (configs with extinction rate <= {MAX_CONFIG_EXTINCTION_RATE})",
                 fontsize=9)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"top{TOP_N}_{metric_key}_timeseries.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"  top {TOP_N} by final {label}:")
    for v, r in top_n:
        print(f"    cfg{r['config']['config_id']:03d} seed{r['seed']}: {v:.4f}")

print("Done.")

# Change