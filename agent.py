import jax
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

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
