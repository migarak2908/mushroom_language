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

    world = world.at[posx, posy, 3].set(signal)

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
                 agent_start_prop,
                 starting_energy,
                 mushroom_prop
                 ):
        self.seed = seed
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.nb_agents = nb_agents
        self.agent_start_prop = agent_start_prop
        self.starting_energy = starting_energy
        self.mushroom_prop = mushroom_prop


    def _reset_fn(self):

        SX = self.grid_x
        SY = self.grid_y
        nb_agents = self.nb_agents

        key = jax.random.key(self.seed)

        key, subkey = jax.random.split(key)
        all_cells = jnp.arange(SX * SY)
        chosen = jax.random.choice(subkey, all_cells, shape=(nb_agents,), replace=False)
        posx = chosen // SY
        posy = chosen % SY

        num_mushroom = round(SX * SY * self.mushroom_prop)

        key, subkey = jax.random.split(key)
        mush_chosen = jax.random.choice(subkey, all_cells, shape=(num_mushroom,), replace=False)
        mushroom_posx = mush_chosen // SY
        mushroom_posy = mush_chosen % SY

        mushroom_type = jnp.where(jnp.arange(num_mushroom) < num_mushroom // 2, 0, 1)
        signal = jnp.zeros(nb_agents)

        world = get_init_world(SX, SY, posx, posy, mushroom_posx, mushroom_posy, mushroom_type, signal)

        alive = jnp.where(jnp.arange(nb_agents) < self.agent_start_prop * nb_agents, 1, 0)
        energy = jnp.where(alive, self.starting_energy, 0)
        age = jnp.zeros(shape=(nb_agents,))
        agents = Agent(alive=alive, posx=posx, posy=posy, energy=energy, age=age)

        return (world, agents)







