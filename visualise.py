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
NB_MUSHROOMS = 100
PERC_RADIUS = 4
INTERVAL = 50  # ms between frames

key = jax.random.key(SEED)
env = MushroomWorld(SEED, SX, SY, NB_AGENTS, NB_MUSHROOMS)
agents, mushrooms = env._reset_fn()

step_fn = eqx.filter_jit(env._step_fn)

fig, ax = plt.subplots(figsize=(7, 7))

mush_scatter = ax.scatter([], [], marker='s', s=30, zorder=2)
agent_scatter = ax.scatter([], [], s=5, color='black', zorder=3)

ax.set_xlim(0, SX)
ax.set_ylim(0, SY)
ax.set_aspect('equal')


def update(_):
    global key, agents, mushrooms
    key, subkey = jax.random.split(key)
    agents, mushrooms = step_fn(subkey, agents, mushrooms, PERC_RADIUS)

    mush_scatter.set_offsets(list(zip(mushrooms.posx.tolist(), mushrooms.posy.tolist())))
    mush_scatter.set_array(mushrooms.type)
    mush_scatter.set_cmap('RdYlGn')
    mush_scatter.set_clim(0, 1)

    agent_scatter.set_offsets(np.column_stack([np.array(agents.posx), np.array(agents.posy)]))

    return mush_scatter, agent_scatter


ani = animation.FuncAnimation(fig, update, interval=INTERVAL, blit=True)
plt.tight_layout()
plt.show()
