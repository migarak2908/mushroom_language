import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NO_SIGNAL = True  # must match the flag used when sweep.py was run
RESULTS_DIR = f"/content/drive/MyDrive/sweep{'_nosig' if NO_SIGNAL else ''}"

BLUE = "#2a78d6"      # approach / env discrimination
RED = "#e34948"       # eat discrimination
AQUA = "#1baf7a"      # population
CRITICAL = "#d03b3b"  # extinction marker
GRID = "#c3c2b7"

with open(os.path.join(RESULTS_DIR, "configs.json")) as f:
    configs = {c["config_id"]: c for c in json.load(f)}

# Group result files by config_id
runs = {}
for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.startswith("cfg") or not fname.endswith(".json") or fname == "configs.json":
        continue
    cfg_id = int(fname[3:6])
    seed = int(fname.split("_seed")[1].split(".")[0])
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        data = json.load(f)
    runs.setdefault(cfg_id, {})[seed] = data


def final_stat(values, frac=0.2):
    """Mean over the last `frac` of a chunk series, matching the sweep script's own summary."""
    vals = [v for v in values[int(len(values) * (1 - frac)):] if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def last_valid(values):
    vals = [v for v in values if not np.isnan(v)]
    return vals[-1] if vals else float("nan")


def best_seed(seed_ids, values):
    """Which seed had the highest value, and what it was. (None, nan) if all nan --
    used to find the best-case outcome across a config's 5 stochastic seeds, since
    the mean can be dragged down by an unlucky seed even when another seed of the
    same config evolved real discrimination."""
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)
    if not valid.any():
        return None, float("nan")
    idx = np.flatnonzero(valid)[np.argmax(values[valid])]
    return seed_ids[idx], float(values[idx])


def argmax_valid(xs, ys):
    """Returns (x, y) at the max valid (non-nan) y, or (nan, nan) if none."""
    ys = np.asarray(ys, dtype=float)
    valid = ~np.isnan(ys)
    if not valid.any():
        return float("nan"), float("nan")
    idx = np.flatnonzero(valid)[np.nanargmax(ys[valid])]
    return xs[idx], float(ys[idx])


# --- Per-config time-series plots + summary rows ---
summary_rows = []

for cfg_id, seeds in sorted(runs.items()):
    cfg = configs[cfg_id]
    fig, (ax_pop, ax_env, ax_probe) = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    seed_ids = []
    seed_final_env, seed_final_approach, seed_final_eat, extinct_flags = [], [], [], []
    seed_max_env, seed_max_approach, seed_max_eat = [], [], []

    for seed, data in sorted(seeds.items()):
        seed_ids.append(seed)
        chunks = np.arange(len(data["chunk_alive"]))
        ax_pop.plot(chunks, data["chunk_alive"], color=AQUA, alpha=0.5, linewidth=1.5)
        ax_env.plot(chunks, data["chunk_disc"], color=BLUE, alpha=0.5, linewidth=1.5)

        probe_chunks = data["probe_chunks"]
        ax_probe.plot(probe_chunks, data["probe_approach_disc"], color=BLUE, alpha=0.6,
                      marker="o", markersize=3, linewidth=1)
        ax_probe.plot(probe_chunks, data["probe_eat_disc"], color=RED, alpha=0.6,
                      marker="o", markersize=3, linewidth=1)

        # mark the peak of each series -- populations often peak mid-run and decline
        # before the end, so "final" alone can understate what a config is capable of
        env_peak_x, env_peak_y = argmax_valid(chunks, data["chunk_disc"])
        ax_env.plot(env_peak_x, env_peak_y, marker="*", color=BLUE, markersize=11,
                    markeredgecolor="#0b0b0b", markeredgewidth=0.4, linestyle="none")

        approach_peak_x, approach_peak_y = argmax_valid(probe_chunks, data["probe_approach_disc"])
        ax_probe.plot(approach_peak_x, approach_peak_y, marker="*", color=BLUE, markersize=11,
                      markeredgecolor="#0b0b0b", markeredgewidth=0.4, linestyle="none")

        eat_peak_x, eat_peak_y = argmax_valid(probe_chunks, data["probe_eat_disc"])
        ax_probe.plot(eat_peak_x, eat_peak_y, marker="*", color=RED, markersize=11,
                      markeredgecolor="#0b0b0b", markeredgewidth=0.4, linestyle="none")

        is_extinct = data["extinction_chunk"] is not None
        extinct_flags.append(is_extinct)
        if is_extinct:
            ax_pop.axvline(data["extinction_chunk"], color=CRITICAL, linestyle="--", linewidth=1)

        seed_final_env.append(final_stat(data["chunk_disc"]))
        seed_final_approach.append(last_valid(data["probe_approach_disc"]))
        seed_final_eat.append(last_valid(data["probe_eat_disc"]))

        seed_max_env.append(env_peak_y)
        seed_max_approach.append(approach_peak_y)
        seed_max_eat.append(eat_peak_y)

    # which seed had the best END-of-run value -- the mean across seeds can hide a
    # config that actually works well, if it just got unlucky on 4 of 5 seeds
    best_env_seed, best_env_val = best_seed(seed_ids, seed_final_env)
    best_approach_seed, best_approach_val = best_seed(seed_ids, seed_final_approach)
    best_eat_seed, best_eat_val = best_seed(seed_ids, seed_final_eat)

    if best_env_seed is not None:
        best_data = seeds[best_env_seed]
        best_chunks = np.arange(len(best_data["chunk_alive"]))
        ax_env.plot(best_chunks, best_data["chunk_disc"], color=BLUE, alpha=1.0, linewidth=2.2, zorder=5)
        ax_env.annotate(f"best seed={best_env_seed} ({best_env_val:.3f})", xy=(0.02, 0.95),
                         xycoords="axes fraction", fontsize=7, color=BLUE, va="top")

    if best_approach_seed is not None:
        best_data = seeds[best_approach_seed]
        ax_probe.plot(best_data["probe_chunks"], best_data["probe_approach_disc"], color=BLUE,
                      alpha=1.0, linewidth=2.2, zorder=5)
        ax_probe.annotate(f"best seed (approach)={best_approach_seed} ({best_approach_val:.3f})",
                           xy=(0.02, 0.95), xycoords="axes fraction", fontsize=7, color=BLUE, va="top")

    if best_eat_seed is not None:
        best_data = seeds[best_eat_seed]
        ax_probe.plot(best_data["probe_chunks"], best_data["probe_eat_disc"], color=RED,
                      alpha=1.0, linewidth=2.2, zorder=5)
        ax_probe.annotate(f"best seed (eat)={best_eat_seed} ({best_eat_val:.3f})",
                           xy=(0.02, 0.87), xycoords="axes fraction", fontsize=7, color=RED, va="top")

    ax_pop.axhline(0, color=GRID, linewidth=0.8)
    ax_pop.set_ylabel("population")

    ax_env.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax_env.set_ylabel("env disc score")

    ax_probe.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax_probe.set_ylabel("probe disc score")
    ax_probe.set_xlabel("chunk (x100 env steps)")
    ax_probe.plot([], [], color=BLUE, marker="o", markersize=3, label="approach")
    ax_probe.plot([], [], color=RED, marker="o", markersize=3, label="eat")
    ax_probe.plot([], [], color=GRID, marker="*", markersize=9, markeredgecolor="#0b0b0b",
                  markeredgewidth=0.4, linestyle="none", label="peak")
    ax_probe.legend(fontsize=7, loc="lower right")

    n_extinct = sum(extinct_flags)
    fig.suptitle(
        f"cfg{cfg_id:03d}  mut={cfg['mutation_std']:.4f} poison={cfg['poison_multiplier']:.2f} "
        f"reprod={cfg['reprod_cost']:.1f} decay={cfg['energy_decay']:.3f}  "
        f"({n_extinct}/{len(seeds)} extinct)",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"cfg{cfg_id:03d}_probed_timeseries.png"), dpi=120)
    plt.close(fig)

    summary_rows.append({
        "config_id": cfg_id,
        **{k: cfg[k] for k in ("mutation_std", "poison_multiplier", "reprod_cost", "energy_decay")},
        "n_seeds": len(seeds),
        "extinction_rate": float(np.mean(extinct_flags)),
        "mean_final_env_disc": float(np.nanmean(seed_final_env)),
        "mean_final_probe_approach_disc": float(np.nanmean(seed_final_approach)),
        "mean_final_probe_eat_disc": float(np.nanmean(seed_final_eat)),
        "mean_max_env_disc": float(np.nanmean(seed_max_env)),
        "mean_max_probe_approach_disc": float(np.nanmean(seed_max_approach)),
        "mean_max_probe_eat_disc": float(np.nanmean(seed_max_eat)),
        "best_seed_final_env_disc": best_env_val,
        "best_seed_final_env_disc_seed": best_env_seed,
        "best_seed_final_probe_approach_disc": best_approach_val,
        "best_seed_final_probe_approach_disc_seed": best_approach_seed,
        "best_seed_final_probe_eat_disc": best_eat_val,
        "best_seed_final_probe_eat_disc_seed": best_eat_seed,
    })

summary = pd.DataFrame(summary_rows).sort_values("config_id")
summary.to_csv(os.path.join(RESULTS_DIR, "config_summary_probed.csv"), index=False)
print(f"Wrote {len(summary)} per-config time-series plots and config_summary_probed.csv")

# --- Overview: does the isolation probe track environment-level discrimination? ---
# Two views: "final" (settled/end-of-run) and "max" (best each config ever reached) --
# a config can peak well above where it settles, so these can tell different stories.
def plot_overview(x_col, y_col, x_label, y_label, title_label, out_name):
    valid = summary.dropna(subset=[x_col, y_col])
    corr = np.corrcoef(valid[x_col], valid[y_col])[0, 1]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(
        valid[x_col], valid[y_col],
        c=valid["extinction_rate"], cmap="Blues", vmin=0, vmax=1,
        s=40, edgecolors="#0b0b0b", linewidths=0.4,
    )
    ax.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax.axvline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_label} (r={corr:.2f})", fontsize=10)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("extinction rate")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, out_name), dpi=120)
    plt.close(fig)

    print(f"{title_label} correlation across {len(valid)} configs: r={corr:.3f}")


plot_overview(
    "mean_final_env_disc", "mean_final_probe_approach_disc",
    "mean final env discrimination", "mean final probe (approach) discrimination",
    "final env vs. final isolation-probe discrimination",
    "all_configs_env_vs_probe_disc_final.png",
)
plot_overview(
    "mean_max_env_disc", "mean_max_probe_approach_disc",
    "mean peak env discrimination", "mean peak probe (approach) discrimination",
    "peak env vs. peak isolation-probe discrimination",
    "all_configs_env_vs_probe_disc_max.png",
)
