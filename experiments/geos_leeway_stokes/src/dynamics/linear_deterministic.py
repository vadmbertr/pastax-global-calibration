from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Real

from pastax.gridded import Gridded
from pastax.utils import meters_to_degrees


class LinearDeterministic(eqx.Module):
    r"""
    Implements the drift as a linear combination of surface currents (geostrophic), leeway velocity, and Stokes drift:
    $$
    \mathbf{u}_a = \mathbf{u}_C + \mathbf{u}_L + C_H \mathbf{u}_H
    $$
    where $\mathbf{u}_H$ is the wave-induced Stokes drift velocity, 
    $\mathbf{u}_L = C_L \mathbf{u}_W$ with $\mathbf{u}_W$ the wind velocity at 10 meters ; 
    $C_L$ and $C_H$ being the drag and wave scalar coefficients, respectively.

    Optimal values for $C_L$ and $C_H$ are expected to lie between 3% and 3.5% ([Minguez et al. (2012)](https://doi.org/10.1007/s00477-011-0548-7)),
    here we consider them as learnable parameters, with initial values equal to 3.25%.

    Attributes
    ----------
    drag : Float[Array, ""] | Float[Array, ""], optional
        The drag coefficient, defaults to `jnp.asarray(0.0325)`.
    wave : Float[Array, ""], optional
        The wave coefficient, defaults to `jnp.asarray(0.0325)`.

    Methods
    -------
    __call__(t, y, args)
        Computes the drift velocity as a linear combination of surface currents (geostrophic), leeway velocity, and Stokes drift.
    """

    drag: Float[Array, ""] = eqx.field(
        default_factory=lambda: 0., 
        converter=lambda x: jnp.asarray(x)
    )
    wave: Float[Array, ""] = eqx.field(
        default_factory=lambda: 0., 
        converter=lambda x: jnp.asarray(x)
    )

    def get_parameters(self) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """
        Returns the drag and wave coefficients.

        Returns
        -------
        tuple[Float[Array, ""], Float[Array, ""]]
            The drag and wave coefficients.
        """
        return {"drag": self.drag, "wave": self.wave}

    def __call__(
        self, t: Real[Array, ""], y: Float[Array, "2"], args: tuple[Gridded, Gridded, Gridded]
    ) -> Float[Array, "2"]:
        """
        Computes the drift velocity as a linear combination of surface currents (geostrophic), leeway velocity, and Stokes drift.

        Parameters
        ----------
        t : Real[Array, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude in degrees).
        args : tuple[Gridded, Gridded, Gridded]
            The [`pastax.gridded.Gridded`][] objects containing the physical fields $\mathbf{u}_C$, $\mathbf{u}_W$ and $\mathbf{u}_H$.

        Returns
        -------
        Float[Array, "2"]
            The Lagrangian drift velocity.
        """
        def interp(field, t, lat, lon):
            uv_dict = field.interp("u", "v", time=t, latitude=lat, longitude=lon)
            return jnp.asarray([uv_dict["v"], uv_dict["u"]])

        latitude, longitude = y

        uc_field, uw_field, uh_field = args
        
        uc_vu = interp(uc_field, t, latitude, longitude)
        uw_vu = interp(uw_field, t, latitude, longitude)
        uh_vu = interp(uh_field, t, latitude, longitude)
        
        dlatlon = uc_vu
        dlatlon += self.drag * uw_vu
        dlatlon += self.wave * uh_vu

        if uc_field.is_spherical_mesh and not uc_field.use_degrees:
            dlatlon = meters_to_degrees(dlatlon, latitude=latitude)

        return dlatlon
    