import jax
import jax.numpy as jnp
import equinox as eqx
from agent import Network

# Initialise a world that is a grid of size X, Y. Populate this grid with n agents at random locations.
# Populate this grid with mushrooms at random locations. Edible mushrooms should cover 2.5% of the total
# grid area. Each agent should be comprised of a neural network (basic - MLP; advanced MLP + LSTM).
# Each agent receives an input of 8 rays (for each direction) with max distance and features of intervening
# objects (16 features x 8 rays = input vector). They have an output of 7 (3 for signal and 4 for movement).

# Create array of valid signals
n = 3
bits = jnp.arange(2**n)
SIGNALS = (bits[:, None] >> jnp.arange(n)[::-1]) & 1
SIGNALS = jnp.concat([SIGNALS, jnp.array([[0.5, 0.5, 0.5]])])


# Create array of valid mushroom features
mushroom_prototype = jnp.array([
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
])

bit_change = jnp.eye(10, dtype=jnp.int32)
FEATURES = mushroom_prototype[:, None, :] ^ bit_change[None, :, :]
FEATURES = jnp.reshape(FEATURES, (-1, 10))
FEATURES = jnp.concat([FEATURES, jnp.full((1, 10), 0.5)])

# Position updates based on direction
TURN = jnp.array([0, -1, 1, 0])
MOVE = jnp.array([0, 0, 0, 1])
DX = jnp.array([0, 1, 0, -1])
DY = jnp.array([1, 0, -1, 0])


class Agents(eqx.Module):
    posx: jnp.ndarray
    posy: jnp.ndarray
    direction: jnp.ndarray
    last_signal: jnp.ndarray
    network: Network
    energy: jnp.ndarray


class Mushrooms(eqx.Module):
    posx: jnp.ndarray
    posy: jnp.ndarray
    type: jnp.ndarray
    features: jnp.ndarray


class MushroomWorld(eqx.Module):
    seed: int
    grid_x: int
    grid_y: int
    nb_agents: int
    nb_mushrooms: int
    energy_start: int


    def _reset_fn(self):

        SX = self.grid_x
        SY = self.grid_y
        nb_agents = self.nb_agents

        # set key
        key = jax.random.key(self.seed)

        # initialise agent positions
        key, subkey = jax.random.split(key)
        all_cells = jnp.arange(SX * SY)
        chosen = jax.random.choice(subkey, all_cells, shape=(nb_agents,), replace=False)
        posx = chosen // SY
        posy = chosen % SY

        # initialise agent directions and empty signal
        key, subkey = jax.random.split(key)
        direction = jax.random.randint(subkey, shape=(nb_agents,), minval=0, maxval=4)
        signal = jnp.zeros(nb_agents, dtype=jnp.int32)

        energy = jnp.full(nb_agents, self.energy_start)

        # initialise mushroom positions, type and features
        num_mushroom = self.nb_mushrooms

        key, subkey = jax.random.split(key)
        mush_chosen = jax.random.choice(subkey, all_cells, shape=(num_mushroom,), replace=False)
        mushroom_posx = mush_chosen // SY
        mushroom_posy = mush_chosen % SY

        mushroom_type = jnp.where(jnp.arange(num_mushroom) < num_mushroom // 2, 0, 1)

        key, subkey1, subkey2 = jax.random.split(key, 3)
        mushroom_features = jnp.where(mushroom_type,
                                      jax.random.randint(subkey1, shape=(num_mushroom,), minval=10, maxval=20),
                                      jax.random.randint(subkey2, shape=(num_mushroom,), minval=0, maxval=10))

        # initialise agent params
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, nb_agents)
        network = eqx.filter_vmap(lambda k: Network(k, input_dim=15, h_size=5, output_dim=5))(keys)

        # create agents and mushrooms data structure
        agents = Agents(posx=posx, posy=posy, direction=direction, last_signal=signal, network=network, energy=energy)
        mushrooms = Mushrooms(posx=mushroom_posx, posy=mushroom_posy, type=mushroom_type, features=mushroom_features)

        return (agents, mushrooms)


    def _compute_obs(self, key, agents, mushrooms, perc_radius):
        key, subkey = jax.random.split(key)

        SX = self.grid_x
        SY = self.grid_y

        # compute distance matrix
        x_diff = agents.posx[:, None] - mushrooms.posx[None, :]
        y_diff = agents.posy[:, None] - mushrooms.posy[None, :]
        x_diff = (x_diff + SX // 2) % SX - SX // 2
        y_diff = (y_diff + SY // 2) % SY - SY // 2

        distance_sq = x_diff ** 2 + y_diff ** 2
        distance_sq = distance_sq + jax.random.uniform(subkey, distance_sq.shape) * 1e-6

        # compute directions
        distance = jnp.sqrt(distance_sq + 1e-8)
        inv_dist = 1.0 / distance
        cos_dir = x_diff * inv_dist
        sin_dir = y_diff * inv_dist

        # find the nearest mushroom for each agent
        nearest_mush = jnp.argmin(distance_sq, axis=1)

        # find input direction for nearest mushrooms per agent
        input_cos = cos_dir[jnp.arange(self.nb_agents), nearest_mush]
        input_sin = sin_dir[jnp.arange(self.nb_agents), nearest_mush]

        # if the agent within perc_radius of nearest mushroom, receive mushroom's perceptual features

        dist_to_mush = distance[jnp.arange(self.nb_agents), nearest_mush]
        features = jnp.where(dist_to_mush <= perc_radius, mushrooms.features[nearest_mush], 20)

        # obtain signals produced in last step
        last_signal = agents.last_signal

        # find the 2 agents closest to each mushroom
        closest_agents = jnp.argsort(distance, axis=0)[:2]
        nearest_agent_per_mush = closest_agents[0, :]
        backup_agent_per_mush = closest_agents[1, :]

        # for each agent find the agent closest to its nearest mushroom and second closest
        agent_at_mush = nearest_agent_per_mush[nearest_mush]
        backup_at_mush = backup_agent_per_mush[nearest_mush]

        # for each agent if the agent closest to its nearest mushroom is itself, use the second-nearest
        agents_idx = jnp.arange(self.nb_agents)
        use_backup = (agents_idx == agent_at_mush)

        signalling_agents = jnp.where(use_backup, backup_at_mush, agent_at_mush)

        # only take signals from agents within the perception radius of the nearest mushroom
        signalling_dist = distance[signalling_agents, nearest_mush]
        signals = jnp.where(signalling_dist <= perc_radius, last_signal[signalling_agents], 8)

        signals = SIGNALS[signals]
        features = FEATURES[features]

        obs = jnp.concat([input_cos[:, None], input_sin[:, None], features, signals], axis=1)

        return obs

    def _compute_update(self, actions, agents, mushrooms):

        # update agent positions (00 = stay, 10 = turn left, 01 = turn right, 11 = move forward)
        movement = actions[:, :2]

        bit_high = movement[:, 0]
        bit_low = movement[:, 1]

        move_idx = 2 * bit_high + bit_low

        turn = TURN[move_idx]
        move = MOVE[move_idx]

        posx = (agents.posx + DX[agents.direction] * move) % self.grid_x
        posy = (agents.posy + DY[agents.direction] * move) % self.grid_y
        direction = (agents.direction + turn) % 4

        # take agent signal bits and convert back to integer for storage
        signal_bits = actions[:, 2:]
        last_signal = signal_bits[:, 0] * 4 + signal_bits[:, 1] * 2 + signal_bits[:, 2]

        # TODO - update agent energy based on mushroom consumption and decay

        # update agents
        agents = Agents(posx=posx, posy=posy, direction=direction, last_signal=last_signal, network=agents.network)

        return agents, mushrooms

    def _step_fn(self, key, agents, mushrooms, perc_radius):

        key, subkey = jax.random.split(key)

        # obtain observations
        obs = self._compute_obs(subkey, agents, mushrooms, perc_radius)

        # compute agent actions given observations
        raw_actions = eqx.filter_vmap(lambda n, o: n(o))(agents.network, obs)
        actions = jnp.round(raw_actions).astype(jnp.int32)

        # update agents and mushrooms given agent actions
        agents, mushrooms = self._compute_update(actions, agents, mushrooms)


        return (agents, mushrooms)


