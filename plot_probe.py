import json
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "/content/drive/MyDrive/probe_results_nosig"

# Group files by config
configs = {}
for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.endswith("_probe.json"):
        continue
    parts = fname.replace("_probe.json", "").split("_")
    cfg_id = f"{parts[0]}_{parts[1]}"
    seed = parts[2]

    with open(os.path.join(RESULTS_DIR, fname)) as f:
        data = json.load(f)

    if cfg_id not in configs:
        configs[cfg_id] = {}
    configs[cfg_id][seed] = data

# Plot one figure per config
for cfg_id, seeds in sorted(configs.items()):
    n_seeds = len(seeds)
    fig, axes = plt.subplots(1, n_seeds, figsize=(4 * n_seeds, 4), sharey=True)
    if n_seeds == 1:
        axes = [axes]

    fig.suptitle(cfg_id, fontsize=12)

    for ax, (seed, data) in zip(axes, sorted(seeds.items())):
        n_alive = data["n_alive"]
        if n_alive == 0:
            ax.text(0.5, 0.5, "extinct", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(seed)
            continue

        approach = np.array(data["approach_disc"])
        eat = np.array(data["eat_disc"])

        ax.hist(approach, bins=40, alpha=0.6, label="approach", color="steelblue")
        ax.hist(eat,      bins=40, alpha=0.6, label="eat",      color="tomato")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(np.mean(approach), color="steelblue", linewidth=1.5)
        ax.axvline(np.mean(eat),      color="tomato",    linewidth=1.5)
        ax.set_title(f"{seed}  (n={n_alive})\nmean={np.mean(approach):.3f}")
        ax.set_xlabel("disc score")
        if ax is axes[0]:
            ax.set_ylabel("agents")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{cfg_id}_probe_dist.png"), dpi=120)
    plt.show()

print("Done.")
