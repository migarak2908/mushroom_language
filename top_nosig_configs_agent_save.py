from mushroom_world import MushroomWorld
import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
import json
import os
import hashlib
from scipy.stats import qmc
import csv
import pandas as pd

CONFIG_DIR = "/content/drive/MyDrive/mushroom_sweep_nosig"
TOP_N = 3



# Fixed params
SX = 100
SY = 100
NB_AGENTS = 2000
MAX_AGENTS = 5000
NB_MUSHROOMS = 100
ENERGY_START = 200.0
MUSHROOM_NUTRITION = 20.0
REPROD_THRESHOLD = 210.
PERC_RADIUS = 10
SHUFFLE_PERIOD = 200
NO_SIGNAL = True
STEPS = 100_000
CHUNK = 100

N_CONFIGS = 100
N_SEEDS = 5
LHS_SEED = 42

RESULTS_DIR = "/content/drive/MyDrive/saved_agents_nosig"
os.makedirs(RESULTS_DIR, exist_ok=True)

def config_hash(cfg):
    s = f"{cfg['mutation_std']:.6f}_{cfg['poison_multiplier']:.4f}_{cfg['reprod_cost']:.4f}_{cfg['energy_decay']:.4f}"
    return hashlib.md5(s.encode()).hexdigest()[:8]


def run_one(cfg, seed):
    key = jax.random.key(seed)
    env = MushroomWorld(
        seed=seed, grid_x=SX, grid_y=SY,
        nb_agents=NB_AGENTS, max_agents=MAX_AGENTS, nb_mushrooms=NB_MUSHROOMS,
        energy_start=ENERGY_START, energy_decay=cfg["energy_decay"],
        mushroom_nutrition=MUSHROOM_NUTRITION,
        reprod_threshold=REPROD_THRESHOLD, reprod_cost=cfg["reprod_cost"],
        shuffle_period=SHUFFLE_PERIOD,
        mutation_std=cfg["mutation_std"],
        poison_multiplier=cfg["poison_multiplier"],
        frozen_baseline=False,
        no_signal=NO_SIGNAL
    )
    agents, mushrooms = env.reset_fn()
    dynamic_agents, static_agents = eqx.partition(agents, eqx.is_array)

    def step(carry, _):
        key, dynamic_agents, mushrooms = carry
        agents = eqx.combine(dynamic_agents, static_agents)
        key, sk1, sk2, sk3, sk4 = jax.random.split(key, 5)

        obs = env._compute_obs(sk1, agents, mushrooms, PERC_RADIUS)
        probs = eqx.filter_vmap(lambda n, o: n(o))(agents.network, obs)
        actions = jax.random.bernoulli(sk2, probs).astype(jnp.int32)

        agents, mushrooms, edible, poisonous = env._compute_update(sk3, actions, agents, mushrooms)
        agents = env._compute_reproduce(sk4, agents, env.mutation_std)

        n_alive = agents.alive.sum()
        mean_energy = jnp.where(n_alive > 0, (agents.energy * agents.alive).sum() / n_alive, 0.0)

        dynamic_agents, _ = eqx.partition(agents, eqx.is_array)
        return (key, dynamic_agents, mushrooms), (n_alive, mean_energy, edible, poisonous)

    @eqx.filter_jit
    def run_chunk(key, dynamic_agents, mushrooms):
        return jax.lax.scan(step, (key, dynamic_agents, mushrooms), None, length=CHUNK)

    chunk_edible, chunk_poisonous, chunk_disc, chunk_alive, chunk_energy = [], [], [], [], []
    extinction_chunk = None

    for c in range(STEPS // CHUNK):
        (key, dynamic_agents, mushrooms), (n_alive, mean_energy, edible, poisonous) = run_chunk(key, dynamic_agents, mushrooms)

        total_e = int(np.array(edible).sum())
        total_p = int(np.array(poisonous).sum())
        total = total_e + total_p
        disc = (total_e - total_p) / total if total > 0 else float('nan')

        chunk_edible.append(total_e)
        chunk_poisonous.append(total_p)
        chunk_disc.append(disc)
        chunk_alive.append(float(np.array(n_alive).mean()))
        chunk_energy.append(float(np.array(mean_energy).mean()))

        # Strict-zero extinction check: alive at end of chunk
        if int(np.array(n_alive)[-1]) == 0:
            extinction_chunk = c
            print(f"  extinction at chunk {c}")
            break
    final_agents = eqx.combine(dynamic_agents, static_agents)

    return {
        "config": cfg,
        "seed": seed,
        "chunk_edible": chunk_edible,
        "chunk_poisonous": chunk_poisonous,
        "chunk_disc": chunk_disc,
        "chunk_alive": chunk_alive,
        "chunk_energy": chunk_energy,
        "extinction_chunk": extinction_chunk,
        "completed_chunks": len(chunk_disc),
    }, final_agents

top_configs = pd.read_csv(os.path.join(CONFIG_DIR, "config_summary.csv"))
top_configs = top_configs[top_configs["extinction_rate"] <= 0.2].sort_values("mean_disc_survivors", ascending=False)
top_configs = top_configs.head(TOP_N)

configs = top_configs[[
    "config_id", "mutation_std", "poison_multiplier", "reprod_cost", "energy_decay"
]].to_dict(orient="records")

# Save the config list once so you have a record
with open(os.path.join(RESULTS_DIR, "configs.json"), "w") as f:
    json.dump(configs, f, indent=2)

total_runs = len(configs) * N_SEEDS
done = 0

for cfg in configs:
    cfg_hash = config_hash(cfg)
    for seed in range(N_SEEDS):
        out_path = os.path.join(RESULTS_DIR, f"cfg{cfg['config_id']:03d}_{cfg_hash}_seed{seed}.json")
        if os.path.exists(out_path):
            done += 1
            continue

        print(f"[{done+1}/{total_runs}] cfg {cfg['config_id']} seed {seed}: "
              f"mut={cfg['mutation_std']:.4f} poison={cfg['poison_multiplier']:.2f} "
              f"reprod={cfg['reprod_cost']:.1f} decay={cfg['energy_decay']:.3f}")

        result, final_agents = run_one(cfg, seed)

        # Save network weights and alive mask
        agents_path = os.path.join(RESULTS_DIR, f"cfg{cfg['config_id']:03d}_{cfg_hash}_seed{seed}_agents.eqx")
        alive_path = os.path.join(RESULTS_DIR, f"cfg{cfg['config_id']:03d}_{cfg_hash}_seed{seed}_alive.npy")
        eqx.tree_serialise_leaves(agents_path, final_agents.network)
        np.save(alive_path, np.array(final_agents.alive))

        final_window = [d for d in result["chunk_disc"][int(len(result["chunk_disc"]) * 0.8):] if not np.isnan(d)]
        final_disc = np.mean(final_window) if final_window else float('nan')
        print(f"  -> final discrimination: {final_disc:.4f}")

        with open(out_path, "w") as f:
            json.dump(result, f)
        done += 1

print(f"\nSweep complete. {total_runs} runs saved to {RESULTS_DIR}")

