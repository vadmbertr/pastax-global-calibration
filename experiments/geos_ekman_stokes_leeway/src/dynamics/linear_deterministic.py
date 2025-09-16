from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Real

from pastax.gridded import Gridded
from pastax.utils import longitude_in_180_180_degrees, meters_to_degrees


def sanitize(arr):
    return jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


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

    Parameters $w_e$, $\theta_e$, $w_s$, and $w_l$ are tunable coefficients, initialized to 1.5%, 45°, 1, and 0.5%, 
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
    depth_integrated_stokes: bool = eqx.field(static=True)
    effective_wavenumber: bool = eqx.field(static=True)
    include_leeway: bool = eqx.field(static=True)

    def get_coefficients(self) -> Float[Array, "4"]:
        """
        Returns the model coefficients.

        Returns
        -------
        Float[Array, "4"]
            The model coefficients in the physical space.
        """
        mu_phy = jnp.exp(self.mu)
        mu_phy = sanitize(mu_phy)

        return mu_phy

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
        
        def geostrophy():
            return sanitize(ug_vu)
        
        def ekman():
            # clockwise in the NH, counter-clockwise in the SH
            ekman_rotation_signed = ekman_rotation * jnp.sign(latitude)
            rotation_matrix = jnp.asarray([
                [jnp.cos(ekman_rotation_signed), -jnp.sin(ekman_rotation_signed)],
                [jnp.sin(ekman_rotation_signed),  jnp.cos(ekman_rotation_signed)]
            ])

            ekman_velocity = rotation_matrix @ uw_vu
            return ekman_scale * sanitize(ekman_velocity)

        def stokes():
            vu0 = uh_vuth[:2]  # Stokes drift at the surface

            if self.depth_integrated_stokes:
                tp, hs = uh_vuth[2:]  # wave peak period and wave significant height
                drogue_depth = 15.

                if not self.effective_wavenumber:
                    g = 9.80665
                    wp = 2 * jnp.pi / tp
                    k = wp ** 2 / g  # peak wavenumber
                else:
                    wp = 2 * jnp.pi / tp
                    u0 = jnp.sqrt(jnp.sum(vu0 ** 2))
                    k = 8 * u0 / (wp * hs ** 2)  # effective monochromatic wavenumber

                # avoid numerical issues
                x = 2 * k * drogue_depth
                factor = jax.lax.cond(
                    x < 1e-6, lambda _: drogue_depth, lambda _: (1 - jnp.exp(-x)) / (2 * k), operand=None
                )
                stokes_drift = vu0 * factor / drogue_depth
            else:
                stokes_drift = vu0

            return stokes_scale * sanitize(stokes_drift)

        def leeway():
            if self.include_leeway:
                factor = leeway_scale
            else:
                factor = 0
            
            return factor * sanitize(uw_vu)

        latitude, longitude = y
        longitude = longitude_in_180_180_degrees(longitude)  # ensure longitude is in [-180, 180] degrees

        uc_field, uw_field, uh_field = args
        
        ug_vu = interp(uc_field, ("v", "u"), t, latitude, longitude)
        uw_vu = interp(uw_field, ("v", "u"), t, latitude, longitude)
        uh_vuth = interp(uh_field, ("v", "u", "t", "h"), t, latitude, longitude)

        ekman_scale, ekman_rotation, stokes_scale, leeway_scale = self.get_coefficients()

        vu = geostrophy() + ekman() + stokes() + leeway()

        if uc_field.is_spherical_mesh and not uc_field.use_degrees:
            vu = meters_to_degrees(vu, latitude=latitude)

        return vu

    @classmethod
    def from_physical_space(
        cls, 
        mu: Float[Array, "4"] | None = None,
        depth_integrated_stokes: bool = True,
        effective_wavenumber: bool = True,
        include_leeway: bool = True
    ):
        """
        Returns a model initialized with the mean coefficients in the log-space, given their physical space 
        counterparts.

        Parameters
        ----------
        mu : Float[Array, "4"], optional
            The mean vector of the model coefficients in the physical space, by default `[1.5%, 45°, 1, 0.5%]`.

        Returns
        -------
        LinearStochastic
            A LinearStochastic model initialized with the given parameters.
        """
        if mu is None:
            mu = jnp.asarray([1.5 / 100, jnp.deg2rad(45.), 1., .5 / 100])
        
        return cls(
            mu=jnp.log(mu),
            depth_integrated_stokes=depth_integrated_stokes,
            effective_wavenumber=effective_wavenumber,
            include_leeway=include_leeway
        )
