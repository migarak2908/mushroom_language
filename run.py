from mushroom_world import MushroomWorld
import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
import wandb

SEED = 0
SX = 100
SY = 100
NB_AGENTS = 2000
MAX_AGENTS = 5000
NB_MUSHROOMS = 100
ENERGY_START = 200.0
ENERGY_DECAY = 0.1
MUSHROOM_NUTRITION = 20.0
REPROD_THRESHOLD = 210.
REPROD_COST = 0.
PERC_RADIUS = 10
STEPS = 100_000
CHUNK = 100

SHUFFLE_PERIOD = 200

key = jax.random.key(SEED)
env = MushroomWorld(SEED, SX, SY, NB_AGENTS, MAX_AGENTS, NB_MUSHROOMS, ENERGY_START, ENERGY_DECAY, MUSHROOM_NUTRITION, REPROD_THRESHOLD, REPROD_COST, SHUFFLE_PERIOD)
agents, mushrooms = env.reset_fn()

dynamic_agents, static_agents = eqx.partition(agents, eqx.is_array)


def step(carry, _):
    key, dynamic_agents, mushrooms = carry
    agents = eqx.combine(dynamic_agents, static_agents)

    key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)

    alive_before = agents.alive.sum()

    obs = env._compute_obs(subkey1, agents, mushrooms, PERC_RADIUS)
    probs, hidden = eqx.filter_vmap(lambda n, o, h: n(o, h))(agents.network, obs, agents.hidden)
    actions = jax.random.bernoulli(subkey2, probs).astype(jnp.int32)
    agents = eqx.tree_at(lambda a: a.hidden, agents, hidden)

    agents, mushrooms, edible_consumed, poisonous_consumed = env._compute_update(subkey3, actions, agents, mushrooms)
    alive_post_update = agents.alive.sum()

    agents = env._compute_reproduce(subkey4, agents)
    alive_post_reproduce = agents.alive.sum()

    deaths = alive_before - alive_post_update
    births = alive_post_reproduce - alive_post_update
    n_alive = alive_post_reproduce
    mean_energy = jnp.where(n_alive > 0, (agents.energy * agents.alive).sum() / n_alive, 0.0)

    alive_mask = agents.alive.astype(bool)
    signal_dist = jnp.array([((agents.last_signal == i) & alive_mask).sum() for i in range(9)])

    dynamic_agents, _ = eqx.partition(agents, eqx.is_array)
    return (key, dynamic_agents, mushrooms), (n_alive, mean_energy, births, deaths, signal_dist, edible_consumed, poisonous_consumed)


@eqx.filter_jit
def run_chunk(key, dynamic_agents, mushrooms):
    return jax.lax.scan(step, (key, dynamic_agents, mushrooms), None, length=CHUNK)


wandb.init(
    project="mushroom-language",
    config=dict(
        seed=SEED, grid_x=SX, grid_y=SY, nb_agents=NB_AGENTS, max_agents=MAX_AGENTS,
        nb_mushrooms=NB_MUSHROOMS, energy_start=ENERGY_START, energy_decay=ENERGY_DECAY,
        mushroom_nutrition=MUSHROOM_NUTRITION, reprod_threshold=REPROD_THRESHOLD,
        reprod_cost=REPROD_COST, perc_radius=PERC_RADIUS, steps=STEPS,
    )
)

signal_labels = [f"signal_{i}" for i in range(8)] + ["signal_silent"]

print("Running simulation...")
for chunk in range(STEPS // CHUNK):
    (key, dynamic_agents, mushrooms), (n_alive, mean_energy, births, deaths, signal_dist, edible_consumed, poisonous_consumed) = run_chunk(key, dynamic_agents, mushrooms)

    n_alive = np.array(n_alive)
    mean_energy = np.array(mean_energy)
    births = np.array(births)
    deaths = np.array(deaths)
    signal_dist = np.array(signal_dist)

    total_edible = int(np.array(edible_consumed).sum())
    total_poisonous = int(np.array(poisonous_consumed).sum())
    disc_score = (total_edible - total_poisonous) / (total_edible + total_poisonous + 1e-8)

    for i in range(CHUNK):
        global_step = chunk * CHUNK + i
        log = {
            "population": int(n_alive[i]),
            "mean_energy": float(mean_energy[i]),
            "births": int(births[i]),
            "deaths": int(deaths[i]),
        }
        for j, label in enumerate(signal_labels):
            log[label] = int(signal_dist[i, j])
        if i == CHUNK - 1:
            log["edible_consumed"] = total_edible
            log["poisonous_consumed"] = total_poisonous
            log["discrimination_score"] = disc_score
        wandb.log(log, step=global_step)

    if chunk % 10 == 0:
        print(f"Step {chunk * CHUNK}/{STEPS} — population: {n_alive[-1]}  disc: {disc_score:.3f}")

wandb.finish()
print("Done.")
