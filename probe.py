from mushroom_world import SIGNALS, MUSH_LIBRARY, DX, DY, TURN, MOVE
from agent import Network
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import os
import json

PROBE_GRID = 20
AGENT_X, AGENT_Y = 10, 10
MUSH_X, MUSH_Y = 10, 18       # distance 8, always within perc_radius=10
INITIAL_DIST = float(MUSH_Y - AGENT_Y)
PROBE_STEPS = 40
PERC_RADIUS = 10
MAX_AGENTS = 5000

AGENTS_DIR = "/content/drive/MyDrive/saved_agents_nosig"
RESULTS_DIR = "/content/drive/MyDrive/probe_results_nosig"
os.makedirs(RESULTS_DIR, exist_ok=True)

NEUTRAL_SIGNAL = SIGNALS[-1]  # [0.5, 0.5, 0.5] — matches no_signal training condition


def run_probe_trial(network, direction, feature_idx):
    """Single probe trial. Returns (approach_score, ate_any)."""
    px = jnp.array(AGENT_X, dtype=jnp.int32)
    py = jnp.array(AGENT_Y, dtype=jnp.int32)
    d  = jnp.array(direction, dtype=jnp.int32)
    mx = jnp.array(MUSH_X, dtype=jnp.int32)
    my = jnp.array(MUSH_Y, dtype=jnp.int32)
    features = MUSH_LIBRARY[feature_idx]

    def dist_to_mush(px, py):
        xd = (px - mx + PROBE_GRID // 2) % PROBE_GRID - PROBE_GRID // 2
        yd = (py - my + PROBE_GRID // 2) % PROBE_GRID - PROBE_GRID // 2
        return jnp.sqrt((xd ** 2 + yd ** 2).astype(jnp.float32) + 1e-8), xd, yd

    def step(carry, _):
        px, py, d = carry

        dist, xd, yd = dist_to_mush(px, py)
        cos_ = xd.astype(jnp.float32) / dist
        sin_ = yd.astype(jnp.float32) / dist
        feat_obs = jnp.where(dist <= PERC_RADIUS, features, MUSH_LIBRARY[20])
        obs = jnp.concat([cos_[None], sin_[None], feat_obs, NEUTRAL_SIGNAL])

        probs = network(obs)
        actions = jnp.round(probs).astype(jnp.int32)

        move_idx = 2 * actions[0] + actions[1]
        new_d  = (d + TURN[move_idx]) % 4
        new_px = (px + DX[d] * MOVE[move_idx]) % PROBE_GRID
        new_py = (py + DY[d] * MOVE[move_idx]) % PROBE_GRID

        new_dist, _, _ = dist_to_mush(new_px, new_py)
        ate = (new_px == mx) & (new_py == my)

        return (new_px, new_py, new_d), (new_dist, ate.astype(jnp.float32))

    _, (distances, ates) = jax.lax.scan(step, (px, py, d), None, length=PROBE_STEPS)

    approach_score = (INITIAL_DIST - jnp.min(distances)) / INITIAL_DIST
    ate_any = (ates.sum() > 0).astype(jnp.float32)
    return approach_score, ate_any


@eqx.filter_jit
def run_trial_all_agents(networks, direction, feature_idx):
    return eqx.filter_vmap(lambda n: run_probe_trial(n, direction, feature_idx))(networks)


def probe_population(networks):
    """Run all trials for all agents. Returns arrays of shape (n_agents,)."""
    edible_approach  = []
    edible_ate       = []
    poison_approach  = []
    poison_ate       = []

    for feat_idx in range(10, 20):       # edible variants
        for direction in range(4):
            a, e = run_trial_all_agents(networks, direction, feat_idx)
            edible_approach.append(np.array(a))
            edible_ate.append(np.array(e))

    for feat_idx in range(0, 10):        # poisonous variants
        for direction in range(4):
            a, e = run_trial_all_agents(networks, direction, feat_idx)
            poison_approach.append(np.array(a))
            poison_ate.append(np.array(e))

    edible_approach  = np.mean(edible_approach,  axis=0)   # (n_agents,)
    edible_ate       = np.mean(edible_ate,        axis=0)
    poison_approach  = np.mean(poison_approach,  axis=0)
    poison_ate       = np.mean(poison_ate,        axis=0)

    return {
        "edible_approach":  edible_approach,
        "edible_eat_rate":  edible_ate,
        "poison_approach":  poison_approach,
        "poison_eat_rate":  poison_ate,
        "approach_disc":    edible_approach - poison_approach,
        "eat_disc":         edible_ate      - poison_ate,
    }


def load_networks(agents_path, alive_path):
    alive = np.load(alive_path).astype(bool)
    dummy_keys = jax.random.split(jax.random.key(0), MAX_AGENTS)
    dummy_networks = eqx.filter_vmap(
        lambda k: Network(k, input_dim=15, h_size=5, output_dim=5)
    )(dummy_keys)
    networks = eqx.tree_deserialise_leaves(agents_path, dummy_networks)
    return networks, alive


# Run probe on all saved agent files
for fname in sorted(os.listdir(AGENTS_DIR)):
    if not fname.endswith("_agents.eqx"):
        continue

    base = fname.replace("_agents.eqx", "")
    alive_path   = os.path.join(AGENTS_DIR, f"{base}_alive.npy")
    out_path     = os.path.join(RESULTS_DIR, f"{base}_probe.json")

    if os.path.exists(out_path):
        continue
    if not os.path.exists(alive_path):
        print(f"Missing alive mask for {base}, skipping")
        continue

    print(f"Probing {base}...")
    networks, alive = load_networks(os.path.join(AGENTS_DIR, fname), alive_path)

    results = probe_population(networks)

    # Filter to alive agents only
    out = {k: v[alive].tolist() for k, v in results.items()}
    out["n_alive"] = int(alive.sum())
    out["mean_approach_disc"] = float(np.mean(results["approach_disc"][alive]))
    out["mean_eat_disc"]      = float(np.mean(results["eat_disc"][alive]))

    with open(out_path, "w") as f:
        json.dump(out, f)

    print(f"  n_alive={out['n_alive']}  approach_disc={out['mean_approach_disc']:.3f}  eat_disc={out['mean_eat_disc']:.3f}")

print("Done.")
