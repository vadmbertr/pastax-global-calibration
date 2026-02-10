import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Float, Key

from src.ec_mlp.data_driven_model import DataDrivenModel


class DriftModel(eqx.Module):
    data_driven_model: DataDrivenModel
    stress_normalization: float
    wind_normalization: float
    delta_t: float

    def __init__(
        self, 
        data_driven_model: DataDrivenModel, 
        stress_normalization: float, 
        wind_normalization: float, 
        delta_t: float = 1.0 * 60.0 * 60.0  # 1 hour in seconds
    ):
        self.data_driven_model = data_driven_model
        self.stress_normalization = stress_normalization
        self.wind_normalization = wind_normalization
        self.delta_t = delta_t
    
    def compute_log_likelihood(
        self,
        ve: Float[jax.Array, ""],
        vn: Float[jax.Array, ""],
        ugos: Float[jax.Array, ""],
        vgos: Float[jax.Array, ""],
        eastward_stress: Float[jax.Array, ""],
        northward_stress: Float[jax.Array, ""],
        eastward_wind: Float[jax.Array, ""],
        northward_wind: Float[jax.Array, ""],
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""]
    ) -> Float[jax.Array, ""]:
        features = self._get_features(month_of_year, lat, lon)

        return self.compute_log_likelihood_from_features(
            ve, vn, ugos, vgos, eastward_stress, northward_stress,eastward_wind, northward_wind, *features
        )
    
    def compute_log_likelihood_from_features(
        self,
        ve: Float[jax.Array, ""],
        vn: Float[jax.Array, ""],
        ugos: Float[jax.Array, ""],
        vgos: Float[jax.Array, ""],
        eastward_stress: Float[jax.Array, ""],
        northward_stress: Float[jax.Array, ""],
        eastward_wind: Float[jax.Array, ""],
        northward_wind: Float[jax.Array, ""],
        month_of_year_cos: Float[jax.Array, ""],
        month_of_year_sin: Float[jax.Array, ""],
        lat_cos: Float[jax.Array, ""],
        lat_sin: Float[jax.Array, ""],
        lon_cos: Float[jax.Array, ""],
        lon_sin: Float[jax.Array, ""]
    ) -> Float[jax.Array, ""]:
        physical_parameters, mdn_parameters = self._get_physical_and_mdn_parameters_from_features(
            month_of_year_cos, month_of_year_sin, lat_cos, lat_sin, lon_cos, lon_sin
        )

        u_est, v_est = self._evaluate_deterministic_term(
            ugos, vgos, eastward_stress, northward_stress, eastward_wind, northward_wind, *physical_parameters
        )

        epsilon_u = ve - u_est  # in m/s
        epsilon_v = vn - v_est  # in m/s

        def gaussian_component_likelihood(
            mu_k: Float[jax.Array, "2"],
            sigma_k: Float[jax.Array, "2"],
            rho_k: Float[jax.Array, ""],
        ):
            sigma_x, sigma_y = sigma_k[0], sigma_k[1]  # in m/s^{1/2}

            logdet = 2 * jnp.log(self.delta_t) + 2 * jnp.log(sigma_x) + 2 * jnp.log(sigma_y) + jnp.log(1 - rho_k ** 2)

            diff_x = (epsilon_u - mu_k[0]) * self.delta_t  # in m
            diff_y = (epsilon_v - mu_k[1]) * self.delta_t  # in m

            quad_term = (
                (diff_x ** 2 / sigma_x ** 2) +  # in s
                (diff_y ** 2 / sigma_y ** 2) - 
                2 * rho_k * diff_x * diff_y / (sigma_x * sigma_y)
            ) / (self.delta_t * (1 - rho_k ** 2))  # nondimensional

            return -(1 / 2) * (logdet + quad_term + 2 * jnp.log(2 * jnp.pi))
        
        pi_k, mu_k, sigma_k, rho_k = mdn_parameters
        l_k = jax.vmap(gaussian_component_likelihood)(mu_k, sigma_k, rho_k)

        return jax.nn.logsumexp(jnp.log(pi_k) + l_k)

    def sample_velocity(
        self,
        ugos: Float[jax.Array, ""],
        vgos: Float[jax.Array, ""],
        eastward_stress: Float[jax.Array, ""],
        northward_stress: Float[jax.Array, ""],
        eastward_wind: Float[jax.Array, ""],
        northward_wind: Float[jax.Array, ""],
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""],
        key: Key[jax.Array, ""]
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
        physical_parameters, mdn_parameters = self.get_physical_and_mdn_parameters(
            month_of_year, lat, lon, to_physical_space=False, in_degrees=False
        )

        u_est, v_est = self._evaluate_deterministic_term(
            ugos, vgos, eastward_stress, northward_stress, eastward_wind, northward_wind, *physical_parameters
        )

        epsilon_u, epsilon_v = self._sample_velocity_residual(*mdn_parameters, key)

        u_sample = u_est + epsilon_u
        v_sample = v_est + epsilon_v

        return u_sample, v_sample

    def estimate_deterministic_velocity(
        self,
        ugos: Float[jax.Array, ""],
        vgos: Float[jax.Array, ""],
        eastward_stress: Float[jax.Array, ""],
        northward_stress: Float[jax.Array, ""],
        eastward_wind: Float[jax.Array, ""],
        northward_wind: Float[jax.Array, ""],
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""]
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
        physical_parameters, _ = self.get_physical_and_mdn_parameters(
            month_of_year, lat, lon, to_physical_space=False, in_degrees=False
        )

        return self._evaluate_deterministic_term(
            ugos, vgos, eastward_stress, northward_stress, eastward_wind, northward_wind, *physical_parameters
        )
    
    def get_mean_residual_velocity(
        self,
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""]
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
        _, mdn_parameters = self.get_physical_and_mdn_parameters(
            month_of_year, lat, lon, to_physical_space=False, in_degrees=False
        )

        pi_k, mu_k, _, _ = mdn_parameters

        return self.get_mean_residual_velocity_from_mdn_parameters(pi_k, mu_k)
    
    def get_mean_residual_velocity_from_mdn_parameters(
        self, pi_k: Float[jax.Array, "n_components"], mu_k: Float[jax.Array, "n_components 2"],
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
        mean_epsilon_u = jnp.sum(pi_k * mu_k[:, 0])  # in m/s
        mean_epsilon_v = jnp.sum(pi_k * mu_k[:, 1])  # in m/s

        return mean_epsilon_u, mean_epsilon_v
    
    def get_mean_residual_displacement(
        self,
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""]
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
        mean_epsilon_u, mean_epsilon_v = self.get_mean_residual_velocity(month_of_year, lat, lon)

        mean_epsilon_x = mean_epsilon_u * self.delta_t  # in m
        mean_epsilon_y = mean_epsilon_v * self.delta_t  # in m

        return mean_epsilon_x, mean_epsilon_y
    
    def get_mode_residual_velocity(
        self,
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""]
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
        _, mdn_parameters = self.get_physical_and_mdn_parameters(
            month_of_year, lat, lon, to_physical_space=False, in_degrees=False
        )

        pi_k, mu_k, _, _ = mdn_parameters

        return self.get_mode_residual_velocity_from_mdn_parameters(pi_k, mu_k)

    def get_mode_residual_velocity_from_mdn_parameters(
        self, pi_k: Float[jax.Array, "n_components"], mu_k: Float[jax.Array, "n_components 2"],
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
        mode_index = jnp.argmax(pi_k)  # mode over components (MAP)
        mode_epsilon_u = mu_k[mode_index, 0]  # in m/s
        mode_epsilon_v = mu_k[mode_index, 1]  # in m/s

        return mode_epsilon_u, mode_epsilon_v

    def get_mode_residual_displacement(
        self,
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""]
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
        mode_epsilon_u, mode_epsilon_v = self.get_mode_residual_velocity(month_of_year, lat, lon)

        mode_epsilon_x = mode_epsilon_u * self.delta_t  # in m
        mode_epsilon_y = mode_epsilon_v * self.delta_t  # in m

        return mode_epsilon_x, mode_epsilon_y
    
    def get_diffusivity_tensor(
        self,
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""]
    ) -> Float[jax.Array, "2 2"]:
        _, mdn_parameters = self.get_physical_and_mdn_parameters(
            month_of_year, lat, lon, to_physical_space=False, in_degrees=False
        )

        return self.get_diffusivity_tensor_from_mdn_parameters(*mdn_parameters)
    
    def get_diffusivity_tensor_from_mdn_parameters(
        self,
        pi_k: Float[jax.Array, "n_components"],
        mu_k: Float[jax.Array, "n_components 2"],
        sigma_k: Float[jax.Array, "n_components 2"],
        rho_k: Float[jax.Array, "n_components"]
    ) -> Float[jax.Array, "2 2"]:
        sigma_x = sigma_k[:, 0]
        sigma_y = sigma_k[:, 1]
        cov_xy = rho_k * sigma_x * sigma_y
        Sigma_k = jnp.stack(
            [jnp.stack([sigma_x ** 2, cov_xy], axis=-1), jnp.stack([cov_xy, sigma_y ** 2], axis=-1)], axis=-2
        )

        exp_cov = jnp.sum(pi_k[:, None, None] * Sigma_k, axis=0)

        mu_bar = jnp.sum(pi_k[:, None] * mu_k, axis=0)
        diffs = mu_k - mu_bar[None, :]
        cov_exp = self.delta_t * jnp.sum(pi_k[:, None, None] * jnp.einsum("ni,nj->nij", diffs, diffs), axis=0)

        total_cov = exp_cov + cov_exp

        K = total_cov / 2
        return K

    def get_crossflow_diffusivity(
        self,
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""]
    ) -> Float[jax.Array, ""]:
        _, mdn_parameters = self.get_physical_and_mdn_parameters(
            month_of_year, lat, lon, to_physical_space=False, in_degrees=False
        )

        return self.get_crossflow_diffusivity_from_mdn_parameters(*mdn_parameters)
    
    def get_crossflow_diffusivity_from_mdn_parameters(
        self,
        pi_k: Float[jax.Array, "n_components"],
        mu_k: Float[jax.Array, "n_components 2"],
        sigma_k: Float[jax.Array, "n_components 2"],
        rho_k: Float[jax.Array, "n_components"]
    ) -> Float[jax.Array, ""]:
        K_tensor = self.get_diffusivity_tensor_from_mdn_parameters(
            pi_k=pi_k,
            mu_k=mu_k,
            sigma_k=sigma_k,
            rho_k=rho_k
        )

        eigvals = jnp.linalg.eigvalsh(K_tensor)

        K_crossflow = eigvals[0]
        return K_crossflow

    def get_physical_and_mdn_parameters(
        self,
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""],
        to_physical_space: bool = True,
        in_degrees: bool = True
    ) -> tuple[
            tuple[
            Float[jax.Array, ""],  # beta_e
            Float[jax.Array, ""],  # theta_e
            Float[jax.Array, ""]  # beta_w
        ], 
        tuple[
            Float[jax.Array, "n_components"],  # pi_k
            Float[jax.Array, "n_components 2"],  # mu_k
            Float[jax.Array, "n_components 2"],  # sigma_k
            Float[jax.Array, "n_components"]  # rho_k
        ]
    ]:
        features = self._get_features(month_of_year, lat, lon)

        physical_parameters, mdn_parameters = self._get_physical_and_mdn_parameters_from_features(*features)

        if to_physical_space:
            physical_parameters = self._transform_physical_parameters_to_physical_space(
                *physical_parameters, to_degrees=in_degrees
            )
        
        return physical_parameters, mdn_parameters
    
    @classmethod
    def _get_features(
        cls,
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""]
    ) -> tuple[
        Float[jax.Array, ""], Float[jax.Array, ""],
        Float[jax.Array, ""], Float[jax.Array, ""],
        Float[jax.Array, ""], Float[jax.Array, ""]
    ]:
        moy_cos = jnp.cos(2 * jnp.pi * month_of_year / 12)
        moy_sin = jnp.sin(2 * jnp.pi * month_of_year / 12)

        lat = jnp.deg2rad(lat)
        lon = jnp.deg2rad(lon)
        lat_cos, lat_sin = jnp.cos(lat), jnp.sin(lat)
        lon_cos, lon_sin = jnp.cos(lon), jnp.sin(lon)

        return moy_cos, moy_sin, lat_cos, lat_sin, lon_cos, lon_sin
    
    def _get_physical_and_mdn_parameters_from_features(
        self,
        month_of_year_cos: Float[jax.Array, ""],
        month_of_year_sin: Float[jax.Array, ""],
        lat_cos: Float[jax.Array, ""],
        lat_sin: Float[jax.Array, ""],
        lon_cos: Float[jax.Array, ""],
        lon_sin: Float[jax.Array, ""]
    ) -> tuple[
            tuple[
            Float[jax.Array, ""], 
            Float[jax.Array, ""], 
            Float[jax.Array, ""]
        ], 
        tuple[
            Float[jax.Array, "n_components"],
            Float[jax.Array, "n_components 2"],
            Float[jax.Array, "n_components 2"],
            Float[jax.Array, "n_components"]
        ]
    ]:
        features = [month_of_year_cos, month_of_year_sin, lat_cos, lat_sin, lon_cos, lon_sin]

        (beta_e, theta_e, beta_w), (pi_k, mu_k, sigma_k, rho_k) = self.data_driven_model(features)

        theta_e = theta_e * jnp.where(lat_sin > 0, -1.0, 1.0)

        mu_k = jnp.reshape(mu_k, (self.data_driven_model.n_components, 2))
        sigma_k = jnp.reshape(sigma_k, (self.data_driven_model.n_components, 2))

        return (beta_e, theta_e, beta_w), (pi_k, mu_k, sigma_k, rho_k)

    def _evaluate_deterministic_term(
        self,
        ugos: Float[jax.Array, ""],
        vgos: Float[jax.Array, ""],
        eastward_stress: Float[jax.Array, ""],
        northward_stress: Float[jax.Array, ""],
        eastward_wind: Float[jax.Array, ""],
        northward_wind: Float[jax.Array, ""],
        beta_e: Float[jax.Array, ""],
        theta_e: Float[jax.Array, ""],
        beta_w: Float[jax.Array, ""]
    ) -> tuple[Float[jax.Array, ""],  Float[jax.Array, ""]]:
        ug = ugos + 1j * vgos
        tau = eastward_stress + 1j * northward_stress
        wind = eastward_wind + 1j * northward_wind

        # normalization trick to ease optimization
        tau_eff = tau / self.stress_normalization
        wind_eff = wind / self.wind_normalization
        beta_e_eff = beta_e * self.stress_normalization
        beta_w_eff = beta_w * self.wind_normalization

        drift_est = ug + beta_e_eff * jnp.exp(1j * theta_e) * tau_eff + beta_w_eff * wind_eff

        return drift_est.real, drift_est.imag

    def _sample_velocity_residual(
        self,
        pi_k: Float[jax.Array, "n_components"],
        mu_k: Float[jax.Array, "n_components 2"],
        sigma_k: Float[jax.Array, "n_components 2"],
        rho_k: Float[jax.Array, "n_components"],
        key: Key[jax.Array, ""]
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
        key, subkey = jrd.split(key)
        component = jrd.choice(subkey, a=pi_k.shape[0], p=pi_k)

        mu_x, mu_y = mu_k[component]  # in m/s
        sigma_x, sigma_y = sigma_k[component]  # in m/s^{1/2}
        rho = rho_k[component]

        key, subkey_x, subkey_y = jrd.split(key, num=3)
        z_x = jrd.normal(subkey_x)
        z_y = jrd.normal(subkey_y)

        u = mu_x + sigma_x / jnp.sqrt(self.delta_t) * z_x  # in m/s
        v = (
            mu_y + 
            rho * sigma_y / jnp.sqrt(self.delta_t) * z_x + 
            jnp.sqrt(1 - rho ** 2) * sigma_y / jnp.sqrt(self.delta_t) * z_y
        )  # in m/s

        return u, v
    
    def _transform_physical_parameters_to_physical_space(
        self,
        beta_e: Float[jax.Array, ""],
        theta_e: Float[jax.Array, ""],
        beta_w: Float[jax.Array, ""],
        to_degrees: bool = True
    ) -> tuple[
        Float[jax.Array, ""], Float[jax.Array, ""],
        Float[jax.Array, ""]
    ]:
        beta_w *= 100
        if to_degrees:
            theta_e = jnp.rad2deg(theta_e)

        return beta_e, theta_e, beta_w
