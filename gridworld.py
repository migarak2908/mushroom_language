import jax
import jax.numpy as jnp
import equinox as eqx


# Initialise a world that is a grid of size X, Y. Populate this grid with n agents at random locations.
# Populate this grid with mushrooms at random locations. Edible mushrooms should cover 2.5% of the total
# grid area. Each agent should be comprised of a neural network (basic - MLP; advanced MLP + LSTM).
# Each agent receives an input of 8 rays (for each direction) with max distance and features of intervening
# objects (16 features x 8 rays = input vector). They have an output of 7 (3 for signal and 4 for movement).

def get_init_world(SX, SY, posx, posy, mushroom_posx, mushroom_posy, mushroom_type, signal):
    world = jnp.zeros((SX, SY, 4))
    world = world.at[posx, posy, 0].set(1)

    world = world.at[mushroom_posx, mushroom_posy, 1].set(1)

    world = world.at[mushroom_posx, mushroom_posy, 2].set(mushroom_type)

    world = world.at[posx, posy, 4].set(signal)

    return world

class Agent(eqx.Module):
    alive: jnp.ndarray
    posx: jnp.ndarray
    posy: jnp.ndarray
    energy: jnp.ndarray
    age: jnp.ndarray


class MushroomWorld(eqx.Module):
    def __init__(self,
                 seed,
                 grid_x,
                 grid_y,
                 nb_agents,
                 start_prop,
                 init_mushrooms,
                 ):
        self.seed = seed
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.nb_agents = nb_agents
        self.start_prop = start_prop
        self.init_mushrooms = init_mushrooms


    def _reset_fn(self):

        SX = self.grid_x
        SY = self.grid_y
        nb_agents = self.nb_agents

        key = jax.random.key(self.seed)
        key, subkey = jax.random.split(key)
        posx = jax.random.randint(subkey, nb_agents, 0, SX - 1)
        key, subkey = jax.random.split(key)
        posy = jax.random.randint(subkey, nb_agents, 0, SY - 1)

        alive = jnp.where(jnp.arange(nb_agents) < self.start_prop * nb_agents, 1, 0)









