from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Real

from pastax.gridded import Gridded
from pastax.utils import longitude_in_180_180_degrees, meters_to_degrees


class LinearDeterministic(eqx.Module):
    r"""
    Implements the drift as a linear combination of surface currents (geostrophic), Ekman currents, Stokes drift, 
    and leeway velocity:
    $$
    \mathbf{u} = \mathbf{u}_g + \mathbf{u}_e + w_s \mathbf{u}_h + w_l \mathbf{u}_w
    $$
    where $\mathbf{u}_g$ is the geostrophic current velocity, $\mathbf{u}_e$ is the Ekman current velocity, 
    $\mathbf{u}_h$ is the wave-induced Stokes drift velocity, and $\mathbf{u}_w$ the wind velocity at 10 meters 
    above the surface.
    Ekman currents are obtained by scaling and rotating the wind velocity at 10 meters above the surface: 
    $\mathbf{u}_e = w_e \exp{-i \theta_e} \mathbf{u}_w$.
    The wave-induced Stokes drift velocity is integrated verticaly from the surface to -15m 
    (this corresponds to the SVP drogue height), giving:
    $\mathbf{u}_h = \mathbf{u}_{h_0} \frac{1 - \exp{2k z_d}}{2k z_d}$, where k is the wave number 
    and $z_d$ is the depth of the drogue (-15m).

    Parameters $w_e$, $\theta_e$, $w_s$, and $w_l$ are tunable coefficients, initialized to 1.5%, 20°, 1, and 0.5%, 
    respectively.

    Attributes
    ----------
    mu: Float[Array, "4"]
        The mean vector of the model coefficients, **in the log-space**.

    Methods
    -------
    __call__(t, y, args)
        Computes the drift velocity as a linear combination of surface currents (geostrophic), Ekman currents, 
        Stokes drift, and leeway velocity.
    """

    mu: Float[Array, "4"] = eqx.field(converter=lambda x: jnp.asarray(x))

    def get_coefficients(self) -> Float[Array, "4"]:
        """
        Returns the model coefficients.

        Returns
        -------
        Float[Array, "4"]
            The model coefficients in the physical space.
        """
        return jnp.exp(self.mu)

    def get_gradients(self) -> Float[Array, "4"]:
        """
        Returns the model coefficients gradients.

        Returns
        -------
        Float[Array, "4"]
            The model coefficients gradients.
        """
        return self.mu

    def __call__(
        self, t: Real[Array, ""], y: Float[Array, "2"], args: tuple[Gridded, Gridded, Gridded]
    ) -> Float[Array, "2"]:
        """
        Computes the drift velocity as a linear combination of surface currents (geostrophic), Ekman currents, 
        Stokes drift, and leeway velocity.

        Parameters
        ----------
        t : Real[Array, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude in degrees).
        args : tuple[Gridded, Gridded, Gridded]
            The [`pastax.gridded.Gridded`][] objects containing the required physical field.

        Returns
        -------
        Float[Array, "2"]
            The Lagrangian drift velocity.
        """
        def interp(field, variables, t, lat, lon):
            uv_dict = field.interp(*variables, time=t, latitude=lat, longitude=lon)
            return jnp.asarray([uv_dict[k] for k in variables])
        
        def sanitize(arr):
            return jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        def geostrophy():
            return sanitize(ug_vu)
        
        def ekman():
            rotation_matrix = jnp.asarray([
                [jnp.cos(ekman_rotation), -jnp.sin(ekman_rotation)],
                [jnp.sin(ekman_rotation),  jnp.cos(ekman_rotation)]
            ])
            ekman_velocity = rotation_matrix @ ug_vu
            return ekman_scale * sanitize(ekman_velocity)

        def stokes():
            vu0 = uh_vut[:-1]  # wave Stokes drift at the surface
            tp = uh_vut[-1]  # wave peak period
            g = 9.80665
            k = (2 * jnp.pi)* 2 / (tp**2 * g)  # wave number
            drogue_depth = 15
            stokes_drift = vu0 * (1 - jnp.exp(-2 * k * drogue_depth)) / (2 * k * drogue_depth)
            return stokes_scale * sanitize(stokes_drift)

        def leeway():
            return leeway_scale * sanitize(uw_vu)

        latitude, longitude = y
        longitude = longitude_in_180_180_degrees(longitude)

        uc_field, uw_field, uh_field = args
        
        ug_vu = interp(uc_field, ("v", "u"), t, latitude, longitude)
        uw_vu = interp(uw_field, ("v", "u"), t, latitude, longitude)
        uh_vut = interp(uh_field, ("v", "u", "t"), t, latitude, longitude)

        ekman_scale, ekman_rotation, stokes_scale, leeway_scale = self.get_coefficients()

        vu = geostrophy() + ekman() + stokes() + leeway()

        if uc_field.is_spherical_mesh and not uc_field.use_degrees:
            vu = meters_to_degrees(vu, latitude=latitude)

        return vu

    @classmethod
    def from_physical_space(cls, mu: Float[Array, "4"] | None = None):
        """
        Returns a model initialized with the mean coefficients in the log-space, given their physical space 
        counterparts.

        Parameters
        ----------
        mu : Float[Array, "4"], optional
            The mean vector of the model coefficients in the physical space, by default `[1.5%, 20°, 1, 0.5%]`.

        Returns
        -------
        LinearStochastic
            A LinearStochastic model initialized with the given parameters.
        """
        if mu is None:
            mu = jnp.asarray([1.5 / 100, jnp.deg2rad(20.), 1., .5 / 100])
        
        return cls(mu=jnp.log(mu))
