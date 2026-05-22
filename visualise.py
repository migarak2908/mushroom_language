from mushroom_world import MushroomWorld
import jax
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
SHUFFLE_PERIOD = 200
INTERVAL = 50  # ms between frames



key = jax.random.key(SEED)
env = MushroomWorld(SEED, SX, SY, NB_AGENTS, MAX_AGENTS, NB_MUSHROOMS, ENERGY_START, ENERGY_DECAY, MUSHROOM_NUTRITION, REPROD_THRESHOLD, REPROD_COST, SHUFFLE_PERIOD)
agents, mushrooms = env.reset_fn()

step_fn = eqx.filter_jit(env.step_fn)

fig, ax = plt.subplots(figsize=(7, 7))

mush_scatter = ax.scatter([], [], marker='s', s=30, zorder=2)
agent_scatter = ax.scatter([], [], s=5, color='black', zorder=3)

ax.set_xlim(0, SX)
ax.set_ylim(0, SY)
ax.set_aspect('equal')


step_count = 0
prev_alive = NB_AGENTS

def update(_):
    global key, agents, mushrooms, step_count, prev_alive
    key, subkey = jax.random.split(key)
    agents, mushrooms = step_fn(subkey, agents, mushrooms, PERC_RADIUS)

    mush_scatter.set_offsets(list(zip(mushrooms.posx.tolist(), mushrooms.posy.tolist())))
    mush_scatter.set_array(mushrooms.type)
    mush_scatter.set_cmap('RdYlGn')
    mush_scatter.set_clim(0, 1)

    alive_mask = np.array(agents.alive, dtype=bool)
    agent_scatter.set_offsets(np.column_stack([np.array(agents.posx)[alive_mask], np.array(agents.posy)[alive_mask]]))

    n_alive = int(alive_mask.sum())
    mean_energy = float(np.array(agents.energy)[alive_mask].mean()) if n_alive > 0 else 0.0
    net_change = n_alive - prev_alive

    print(f"\rstep {step_count:>6}  |  pop {n_alive:>5}  |  energy {mean_energy:>8.1f}  |  net {net_change:>+5}", end="", flush=True)

    step_count += 1
    prev_alive = n_alive

    return mush_scatter, agent_scatter


ani = animation.FuncAnimation(fig, update, interval=INTERVAL, blit=True)
plt.tight_layout()
plt.show()
