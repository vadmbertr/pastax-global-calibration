import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float

from src.commons.mlp import MLP


class DataDrivenModel(eqx.Module):
    trunk: MLP
    physical_head: MLP
    mdn_head: MLP
    n_components: int
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

    def __init__(self, trunk: MLP, physical_head: MLP, mdn_head: MLP):
        trunk_output_dim = trunk.layers[-2].out_features  # -2 because last layer is an activation "layer"
        physical_head_input_dim = physical_head.layers[0].in_features
        mdn_head_input_dim = mdn_head.layers[0].in_features

        assert trunk_output_dim == physical_head_input_dim == mdn_head_input_dim, (
            f"Trunk output dimension ({trunk_output_dim}) must match "
            f"physical head input dimension ({physical_head_input_dim}) and "
            f"MDN head input dimension ({mdn_head_input_dim})."
        )

        physical_head_output_dim = physical_head.layers[-2].out_features
        mdn_head_output_dim = mdn_head.layers[-2].out_features

        assert physical_head_output_dim == 3, (
            f"Physical head output dimension must be 3, got {physical_head_output_dim}."
        )

        self.trunk = trunk
        self.physical_head = physical_head
        self.mdn_head = mdn_head

        self.n_components = int(mdn_head_output_dim // (1 + 2 + 3))  # pi + (mu_x, mu_y) + (sigma_x, sigma_y, rho)

        self.beta_e_clim = 0.3
        self.beta_e_max = 5.0
        self.z_beta_e_bg = jsp.special.logit(self.beta_e_clim / self.beta_e_max).item()
        self.beta_e_scale = 3.0

        self.theta_e_clim = jnp.radians(75).item()
        self.theta_e_min = jnp.radians(45).item()
        self.theta_e_max = jnp.radians(135).item()
        self.z_theta_e_bg = jsp.special.logit(
            (self.theta_e_clim - self.theta_e_min) / (self.theta_e_max - self.theta_e_min)
        ).item()
        self.theta_e_scale = 1.5

        self.beta_w_clim = 0.37 / 100
        self.beta_w_max = 10.0 / 100
        self.z_beta_w_bg = jsp.special.logit(self.beta_w_clim / self.beta_w_max).item()
        self.beta_w_scale = 3.0

    def _physical_final_activation(
        self, output: Float[jax.Array, "3"],
    ) -> tuple[
        Float[jax.Array, ""], 
        Float[jax.Array, ""], 
        Float[jax.Array, ""]
    ]:
        beta_e_r, theta_e_r, beta_w_r = output

        beta_e = self.beta_e_max * jax.nn.sigmoid(self.z_beta_e_bg + self.beta_e_scale * jnp.tanh(beta_e_r))
        theta_e = self.theta_e_min + (self.theta_e_max - self.theta_e_min) * jax.nn.sigmoid(
            self.z_theta_e_bg + self.theta_e_scale * jnp.tanh(theta_e_r)
        )
        beta_w = self.beta_w_max * jax.nn.sigmoid(self.z_beta_w_bg + self.beta_w_scale * jnp.tanh(beta_w_r))

        return beta_e, theta_e, beta_w
    
    def _mdn_final_activation(
        self, output: Float[jax.Array, ""]
    ) -> tuple[
        Float[jax.Array, "n_components"],
        Float[jax.Array, "n_components * 2"],
        Float[jax.Array, "n_components * 2"],
        Float[jax.Array, "n_components * 1"]
    ]:
        # MDN: \sum_{i=1}^{n} pi_i * N(\mu_i \Delta t, Sigma_i \Delta t)
        # where \Sigma_i = [[\sigma_{i,x}^2, \rho*\sigma_{i,x}*\sigma_{i,y}], [\rho*\sigma_{i,x}*\sigma_{i,y}, \sigma_{i,y}^2]]
        # and \sum_{i=1}^{n} \pi_i = 1, \pi_i >= 0
        # \mu_i are velocities in [m/s]
        # \Sigma_i entries are variances/covariances in [m^2/s] (\sigma_i are in [m/s^(1/2)])

        pi_logits = output[: self.n_components]
        mus = output[self.n_components : self.n_components * 3]
        log_sigmas = output[self.n_components * 3 : self.n_components * 5]
        rhos = output[self.n_components * 5 :]

        eps = 1e-8
        pis = jax.nn.softmax(pi_logits + eps)  # + eps to prevent numerical issues with zero probabilities
        sigmas = jnp.exp(log_sigmas) + eps  # + eps to prevent numerical issues with zero variances
        rhos = jnp.tanh(rhos) * 0.99  # * 0.99 to prevent numerical issues with correlation values close to 1

        return pis, mus, sigmas, rhos

    def __call__(
        self, 
        features: tuple[
            Float[jax.Array, ""], Float[jax.Array, ""],
            Float[jax.Array, ""], Float[jax.Array, ""],
            Float[jax.Array, ""], Float[jax.Array, ""]
        ],
    ) -> tuple[
            tuple[
            Float[jax.Array, ""], 
            Float[jax.Array, ""], 
            Float[jax.Array, ""]
        ], 
        tuple[
            Float[jax.Array, "n_components"],
            Float[jax.Array, "n_components * 2"],
            Float[jax.Array, "n_components * 2"],
            Float[jax.Array, "n_components"]
        ]
    ]:
        features = jnp.stack(features, axis=-1)
        trunk_output = self.trunk(features)

        physical_output = self.physical_head(trunk_output)
        mdn_output = self.mdn_head(trunk_output)

        physical_parameters = self._physical_final_activation(physical_output)
        mdn_parameters = self._mdn_final_activation(mdn_output)

        return physical_parameters, mdn_parameters 
