import jax
import jax.numpy as jnp
import equinox as eqx



class Network(eqx.Module):
    layers: list

    def __init__(self, key, input_dim, h_size, output_dim):

        key1, key2 = jax.random.split(key)

        self.layers = [
            eqx.nn.Linear(input_dim, h_size, key=key1),
            jax.nn.sigmoid,
            eqx.nn.Linear(h_size, output_dim, key=key2),
            jax.nn.sigmoid
        ]

    def __call__(self, x, hidden):
        for layer in self.layers:
            x = layer(x)

        return x, hidden

    def init_hidden(self):
        return jnp.zeros(0)


class Recurrent_Network(eqx.Module):
    i2h: eqx.nn.Linear
    h2h: eqx.nn.Linear
    h2o: eqx.nn.Linear
    h_size: int = eqx.field(static=True)

    def __init__(self, key, input_dim, h_size, output_dim):

        k1, k2, k3 = jax.random.split(key, 3)
        self.i2h = eqx.nn.Linear(input_dim, h_size, key=k1)
        self.h2h = eqx.nn.Linear(h_size, h_size, key=k2, use_bias=False)
        self.h2o = eqx.nn.Linear(h_size, output_dim, key=k3)
        self.h_size = h_size

    def __call__(self, x, hidden):
        hidden = jax.nn.tanh(self.i2h(x) + self.h2h(hidden))
        output = jax.nn.sigmoid(self.h2o(hidden))
        return output, hidden

    def init_hidden(self):
        return jnp.zeros(self.h_size)