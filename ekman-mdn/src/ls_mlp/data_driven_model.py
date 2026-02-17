import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float

from src.commons.mlp import MLP


class DataDrivenModel(eqx.Module):
    mlp: MLP
    beta_e_clim: float
    beta_e_max: float
    z_beta_e_bg: float
    beta_e_scale: float
    theta_e_clim: float
    theta_e_min: float
    theta_e_max: float
    z_theta_e_bg: float
    theta_e_scale: float
    beta_w_clim: float
    beta_w_max: float
    z_beta_w_bg: float
    beta_w_scale: float

    def __init__(self, mlp: MLP):
        assert mlp.layers[-2].out_features == 5, (f"MLP output dimension must be 5, got {mlp.layers[-2]}.")

        self.mlp = mlp

        self.beta_e_clim = 0.3
        self.beta_e_max = 5.0
        self.z_beta_e_bg = jsp.special.logit(self.beta_e_clim / self.beta_e_max).item()
        self.beta_e_scale = 3.0

        self.theta_e_clim = jnp.radians(75).item()
        self.theta_e_min = jnp.radians(40).item()
        self.theta_e_max = jnp.radians(100).item()
        self.z_theta_e_bg = jsp.special.logit(
            (self.theta_e_clim - self.theta_e_min) / (self.theta_e_max - self.theta_e_min)
        ).item()
        self.theta_e_scale = 1.5

        self.beta_w_clim = 0.37 / 100
        self.beta_w_max = 10.0 / 100
        self.z_beta_w_bg = jsp.special.logit(self.beta_w_clim / self.beta_w_max).item()
        self.beta_w_scale = 3.0

    def _final_custom_activation(
        self, output: Float[jax.Array, "5"]
    ) -> tuple[
        Float[jax.Array, ""], 
        Float[jax.Array, ""], 
        Float[jax.Array, ""], 
        Float[jax.Array, ""], 
        Float[jax.Array, ""]
    ]:
        beta_e_r, theta_e_r, beta_w_r, u_eps, v_eps = output

        beta_e = self.beta_e_max * jax.nn.sigmoid(self.z_beta_e_bg + self.beta_e_scale * jnp.tanh(beta_e_r))
        theta_e = self.theta_e_min + (self.theta_e_max - self.theta_e_min) * jax.nn.sigmoid(
            self.z_theta_e_bg + self.theta_e_scale * jnp.tanh(theta_e_r)
        )
        beta_w = self.beta_w_max * jax.nn.sigmoid(self.z_beta_w_bg + self.beta_w_scale * jnp.tanh(beta_w_r))

        return beta_e, theta_e, beta_w, u_eps, v_eps

    def __call__(
        self, 
        features: tuple[
            Float[jax.Array, ""], Float[jax.Array, ""],
            Float[jax.Array, ""], Float[jax.Array, ""],
            Float[jax.Array, ""], Float[jax.Array, ""]
        ]
    ) -> tuple[
        Float[jax.Array, ""], 
        Float[jax.Array, ""], 
        Float[jax.Array, ""], 
        Float[jax.Array, ""], 
        Float[jax.Array, ""]
    ]:
        features = jnp.stack(features, axis=-1)
        output = self.mlp(features)
        return self._final_custom_activation(output)
