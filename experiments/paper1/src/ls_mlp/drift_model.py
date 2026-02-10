import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Complex, Float

from src.ls_mlp.data_driven_model import DataDrivenModel


class DriftModel(eqx.Module):
    data_driven_model: DataDrivenModel
    stress_normalization: float
    wind_normalization: float

    def __init__(
        self, 
        data_driven_model: DataDrivenModel, 
        stress_normalization: float, 
        wind_normalization: float, 
    ):
        self.data_driven_model = data_driven_model
        self.stress_normalization = stress_normalization
        self.wind_normalization = wind_normalization

    def estimate_velocity(
        self,
        month_of_year: Float[jax.Array, ""],
        lat: Float[jax.Array, ""],
        lon: Float[jax.Array, ""],
        ugos: Float[jax.Array, ""],
        vgos: Float[jax.Array, ""],
        eastward_stress: Float[jax.Array, ""],
        northward_stress: Float[jax.Array, ""],
        eastward_wind: Float[jax.Array, ""],
        northward_wind: Float[jax.Array, ""]
    ) -> Complex[jax.Array, ""]:
        features = self._get_features(month_of_year, lat, lon)
        
        return self(ugos, vgos, eastward_stress, northward_stress, eastward_wind, northward_wind, *features)
    
    def get_physical_parameters(
        self, 
        month_of_year: Float[jax.Array, ""], 
        lat: Float[jax.Array, ""], 
        lon: Float[jax.Array, ""], 
        in_degrees: bool = True
    ) -> tuple[
        Float[jax.Array, ""], Float[jax.Array, ""],
        Float[jax.Array, ""],
        Float[jax.Array, ""], Float[jax.Array, ""]
    ]:
        features = self._get_features(month_of_year, lat, lon)

        beta_e, theta_e, beta_w, u_eps, v_eps = self._get_drift_parameters_from_features(*features)

        beta_e, theta_e, beta_w = self._transform_parameters_to_physical_space(
            beta_e, theta_e, beta_w, to_degrees=in_degrees
        )

        return beta_e, theta_e, beta_w, u_eps, v_eps
    
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
    
    def _get_drift_parameters_from_features(
        self,
        month_of_year_cos: Float[jax.Array, ""],
        month_of_year_sin: Float[jax.Array, ""],
        lat_cos: Float[jax.Array, ""],
        lat_sin: Float[jax.Array, ""],
        lon_cos: Float[jax.Array, ""],
        lon_sin: Float[jax.Array, ""]
    ) -> tuple[
        Float[jax.Array, ""], Float[jax.Array, ""],
        Float[jax.Array, ""],
        Float[jax.Array, ""], Float[jax.Array, ""]
    ]:
        features = [month_of_year_cos, month_of_year_sin, lat_cos, lat_sin, lon_cos, lon_sin]

        beta_e, theta_e, beta_w, u_eps, v_eps = self.data_driven_model(features)
        theta_e = theta_e * jnp.where(lat_sin > 0, -1.0, 1.0)

        return beta_e, theta_e, beta_w, u_eps, v_eps
    
    def _transform_parameters_to_physical_space(
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

    def __call__(
        self,
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
    ) -> Complex[jax.Array, ""]:
        ug = ugos + 1j * vgos
        tau = eastward_stress + 1j * northward_stress
        wind = eastward_wind + 1j * northward_wind

        beta_e, theta_e, beta_w, u_eps, v_eps = self._get_drift_parameters_from_features(
            month_of_year_cos, month_of_year_sin, lat_cos, lat_sin, lon_cos, lon_sin
        )
        eps = u_eps + 1j * v_eps

        # normalization trick to ease optimization
        tau_eff = tau / self.stress_normalization
        wind_eff = wind / self.wind_normalization
        beta_e_eff = beta_e * self.stress_normalization
        beta_w_eff = beta_w * self.wind_normalization

        uv_est = ug + beta_e_eff * jnp.exp(1j * theta_e) * tau_eff + beta_w_eff * wind_eff + eps

        return uv_est
