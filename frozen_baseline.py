from mushroom_world import MushroomWorld
import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
import wandb
import json
import os

# Config (fixed across seeds)
SX = 100
SY = 100
NB_AGENTS = 5000        # start at MAX_AGENTS for frozen baseline
MAX_AGENTS = 5000
NB_MUSHROOMS = 100
ENERGY_START = 200.0
ENERGY_DECAY = 0.1
MUSHROOM_NUTRITION = 20.0
POISON_MULTIPLIER = -1
REPROD_THRESHOLD = 210.
REPROD_COST = 0.
PERC_RADIUS = 10
STEPS = 100_000
CHUNK = 100
SHUFFLE_PERIOD = 200
FROZEN_BASELINE = True

N_SEEDS = 25
RESULTS_DIR = "/content/drive/MyDrive/mushroom_baseline"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_one_seed(seed):
    key = jax.random.key(seed)
    env = MushroomWorld(seed, SX, SY, NB_AGENTS, MAX_AGENTS, NB_MUSHROOMS,
                        ENERGY_START, ENERGY_DECAY, MUSHROOM_NUTRITION, POISON_MULTIPLIER,
                        REPROD_THRESHOLD, REPROD_COST, SHUFFLE_PERIOD, FROZEN_BASELINE)
    agents, mushrooms = env.reset_fn()
    dynamic_agents, static_agents = eqx.partition(agents, eqx.is_array)

    def step(carry, _):
        key, dynamic_agents, mushrooms = carry
        agents = eqx.combine(dynamic_agents, static_agents)

        key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)

        alive_before = agents.alive.sum()
        obs = env._compute_obs(subkey1, agents, mushrooms, PERC_RADIUS)
        probs = eqx.filter_vmap(lambda n, o: n(o))(agents.network, obs)
        actions = jax.random.bernoulli(subkey2, probs).astype(jnp.int32)
        agents, mushrooms, edible_consumed, poisonous_consumed = env._compute_update(subkey3, actions, agents, mushrooms)
        alive_post_update = agents.alive.sum()
        agents = env._compute_respawn(subkey4, agents)
        alive_post_reproduce = agents.alive.sum()

        deaths = alive_before - alive_post_update
        births = alive_post_reproduce - alive_post_update
        n_alive = alive_post_reproduce
        mean_energy = jnp.where(n_alive > 0, (agents.energy * agents.alive).sum() / n_alive, 0.0)

        dynamic_agents, _ = eqx.partition(agents, eqx.is_array)
        return (key, dynamic_agents, mushrooms), (n_alive, mean_energy, births, deaths, edible_consumed, poisonous_consumed)

    @eqx.filter_jit
    def run_chunk(key, dynamic_agents, mushrooms):
        return jax.lax.scan(step, (key, dynamic_agents, mushrooms), None, length=CHUNK)

    chunk_edible = []
    chunk_poisonous = []
    chunk_disc = []
    chunk_alive = []
    chunk_energy = []

    for chunk in range(STEPS // CHUNK):
        (key, dynamic_agents, mushrooms), (n_alive, mean_energy, _, _, edible_consumed, poisonous_consumed) = run_chunk(key, dynamic_agents, mushrooms)

        total_edible = int(np.array(edible_consumed).sum())
        total_poisonous = int(np.array(poisonous_consumed).sum())
        total = total_edible + total_poisonous
        disc_score = (total_edible - total_poisonous) / total if total > 0 else np.nan

        chunk_edible.append(total_edible)
        chunk_poisonous.append(total_poisonous)
        chunk_disc.append(disc_score)
        chunk_alive.append(float(np.array(n_alive).mean()))
        chunk_energy.append(float(np.array(mean_energy).mean()))

        if chunk % 50 == 0:
            print(f"  seed {seed}, chunk {chunk}/{STEPS // CHUNK}, disc={disc_score:.3f}")

    return {
        "seed": seed,
        "chunk_edible": chunk_edible,
        "chunk_poisonous": chunk_poisonous,
        "chunk_disc": chunk_disc,
        "chunk_alive": chunk_alive,
        "chunk_energy": chunk_energy,
    }


# Run all seeds, skip completed
for seed in range(N_SEEDS):
    out_path = os.path.join(RESULTS_DIR, f"seed_{seed:03d}.json")
    if os.path.exists(out_path):
        print(f"Seed {seed} already done, skipping.")
        continue

    print(f"Running seed {seed}...")
    result = run_one_seed(seed)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"Seed {seed} saved.")


# Aggregate
print("\nAggregating results...")
all_results = []
for seed in range(N_SEEDS):
    out_path = os.path.join(RESULTS_DIR, f"seed_{seed:03d}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results.append(json.load(f))

# Final-window discrimination per seed
n_chunks = STEPS // CHUNK
final_window_start = int(n_chunks * 0.8)
final_disc_per_seed = []
for r in all_results:
    final_chunks = [d for d in r["chunk_disc"][final_window_start:] if not np.isnan(d)]
    if final_chunks:
        final_disc_per_seed.append(np.mean(final_chunks))

final_disc_per_seed = np.array(final_disc_per_seed)

print(f"\nBaseline summary across {len(final_disc_per_seed)} seeds:")
print(f"  Mean final discrimination: {final_disc_per_seed.mean():.4f}")
print(f"  Std:                       {final_disc_per_seed.std():.4f}")
print(f"  5th percentile:            {np.percentile(final_disc_per_seed, 5):.4f}")
print(f"  95th percentile:           {np.percentile(final_disc_per_seed, 95):.4f}")
print(f"  Min:                       {final_disc_per_seed.min():.4f}")
print(f"  Max:                       {final_disc_per_seed.max():.4f}")

# Save summary
summary = {
    "n_seeds": int(len(final_disc_per_seed)),
    "final_window_start_chunk": final_window_start,
    "per_seed_final_disc": final_disc_per_seed.tolist(),
    "mean": float(final_disc_per_seed.mean()),
    "std": float(final_disc_per_seed.std()),
    "p5": float(np.percentile(final_disc_per_seed, 5)),
    "p95": float(np.percentile(final_disc_per_seed, 95)),
    "min": float(final_disc_per_seed.min()),
    "max": float(final_disc_per_seed.max()),
}
with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {RESULTS_DIR}/summary.json")