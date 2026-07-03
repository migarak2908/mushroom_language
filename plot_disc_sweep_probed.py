import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NO_SIGNAL = True  # must match the flag used when disc_score_sweep_probed.py was run
RESULTS_DIR = f"/content/drive/MyDrive/mushroom_sweep_probed{'_nosig' if NO_SIGNAL else ''}"

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


# --- Per-config time-series plots + summary rows ---
summary_rows = []

for cfg_id, seeds in sorted(runs.items()):
    cfg = configs[cfg_id]
    fig, (ax_pop, ax_env, ax_probe) = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    seed_final_env, seed_final_approach, seed_final_eat, extinct_flags = [], [], [], []

    for seed, data in sorted(seeds.items()):
        chunks = np.arange(len(data["chunk_alive"]))
        ax_pop.plot(chunks, data["chunk_alive"], color=AQUA, alpha=0.5, linewidth=1.5)
        ax_env.plot(chunks, data["chunk_disc"], color=BLUE, alpha=0.5, linewidth=1.5)

        probe_chunks = data["probe_chunks"]
        ax_probe.plot(probe_chunks, data["probe_approach_disc"], color=BLUE, alpha=0.6,
                      marker="o", markersize=3, linewidth=1)
        ax_probe.plot(probe_chunks, data["probe_eat_disc"], color=RED, alpha=0.6,
                      marker="o", markersize=3, linewidth=1)

        is_extinct = data["extinction_chunk"] is not None
        extinct_flags.append(is_extinct)
        if is_extinct:
            ax_pop.axvline(data["extinction_chunk"], color=CRITICAL, linestyle="--", linewidth=1)

        seed_final_env.append(final_stat(data["chunk_disc"]))
        seed_final_approach.append(last_valid(data["probe_approach_disc"]))
        seed_final_eat.append(last_valid(data["probe_eat_disc"]))

    ax_pop.axhline(0, color=GRID, linewidth=0.8)
    ax_pop.set_ylabel("population")

    ax_env.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax_env.set_ylabel("env disc score")

    ax_probe.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax_probe.set_ylabel("probe disc score")
    ax_probe.set_xlabel("chunk (x100 env steps)")
    ax_probe.plot([], [], color=BLUE, marker="o", markersize=3, label="approach")
    ax_probe.plot([], [], color=RED, marker="o", markersize=3, label="eat")
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
    })

summary = pd.DataFrame(summary_rows).sort_values("config_id")
summary.to_csv(os.path.join(RESULTS_DIR, "config_summary_probed.csv"), index=False)
print(f"Wrote {len(summary)} per-config time-series plots and config_summary_probed.csv")

# --- Overview: does the isolation probe track environment-level discrimination? ---
valid = summary.dropna(subset=["mean_final_env_disc", "mean_final_probe_approach_disc"])
corr = np.corrcoef(valid["mean_final_env_disc"], valid["mean_final_probe_approach_disc"])[0, 1]

fig, ax = plt.subplots(figsize=(5.5, 5))
sc = ax.scatter(
    valid["mean_final_env_disc"], valid["mean_final_probe_approach_disc"],
    c=valid["extinction_rate"], cmap="Blues", vmin=0, vmax=1,
    s=40, edgecolors="#0b0b0b", linewidths=0.4,
)
ax.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
ax.axvline(0, color=GRID, linewidth=0.8, linestyle="--")
ax.set_xlabel("mean final env discrimination")
ax.set_ylabel("mean final probe (approach) discrimination")
ax.set_title(f"env vs. isolation-probe discrimination (r={corr:.2f})", fontsize=10)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("extinction rate")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "all_configs_env_vs_probe_disc.png"), dpi=120)
plt.close(fig)

print(f"env vs probe discrimination correlation across {len(valid)} configs: r={corr:.3f}")
