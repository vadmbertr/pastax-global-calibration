from collections.abc import Iterable
from typing import Callable

import equinox as eqx
import jax
import jax.random as jrd
from jaxtyping import Float, Key


def identity(x: Float[jax.Array, "..."]) -> Float[jax.Array, "..."]:
    return x


class MLP(eqx.Module):
    layers: list[eqx.Module]

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_layers_size: list[int],
        activations: Callable | list[Callable] = jax.nn.gelu,
        final_activation: Callable = identity,
        use_biases: bool | list[bool] = True,
        use_final_bias: bool = True,
        add_layer_norm: bool = True,
        key: Key[jax.Array, ""] | None = None,
    ):
        if key is None:
            key = jrd.key(0)

        if not isinstance(activations, Iterable):
            activations = [activations] * len(hidden_layers_size)
        if not isinstance(use_biases, Iterable):
            use_biases = [use_biases] * len(hidden_layers_size)

        assert len(hidden_layers_size) == len(activations) == len(use_biases), (
            "hidden_layers_size, activations, and use_biases must have the same length."
        )
        
        layers = []
        in_features = in_size

        for width_size, activation, use_bias in zip(hidden_layers_size, activations, use_biases):
            out_features = width_size

            key, subkey = jrd.split(key)
            linear = eqx.nn.Linear(in_features=in_features, out_features=out_features, use_bias=use_bias, key=subkey)
            layers.append(linear)

            if add_layer_norm:
                layer_norm = eqx.nn.LayerNorm(shape=out_features)
                layers.append(layer_norm)

            layers.append(activation)

            in_features = out_features

        key, subkey = jrd.split(key)
        final_layer = eqx.nn.Linear(
            in_features=in_features,
            out_features=out_size,
            use_bias=use_final_bias,
            key=subkey,
        )
        layers.append(final_layer)
        layers.append(final_activation)

        self.layers = layers

    def __call__(self, x: Float[jax.Array, "in_size"]) -> Float[jax.Array, "out_size"]:
        for layer in self.layers:
            x = layer(x)
        return x
