from mushroom_world import MushroomWorld
import jax.numpy as jnp
import jax

SEED = 0
SX = 100
SY = 100
NB_AGENTS = 1000
NB_MUSHROOMS = 100
STEPS = 100_000
PERC_RADIUS = 4

key = jax.random.key(SEED)

env = MushroomWorld(SEED, SX, SY, NB_AGENTS, NB_MUSHROOMS)


agents, mushrooms = env._reset_fn()

def step(carry, _):
    key, agents, mushrooms = carry
    key, subkey = jax.random.split(key)
    agents, mushrooms = env._step_fn(subkey, agents, mushrooms, PERC_RADIUS)
    return (key, agents, mushrooms), None

(key, agents, mushrooms), _ = jax.lax.scan(step, (key, agents, mushrooms), None, length=STEPS)

