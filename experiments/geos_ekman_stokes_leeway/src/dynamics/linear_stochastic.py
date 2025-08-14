from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Real

from pastax.gridded import Gridded
from pastax.utils import longitude_in_180_180_degrees, meters_to_degrees


def softplus_inverse(y: Real[Array, "..."]) -> Real[Array, "..."]:
    return jnp.log(jnp.expm1(y))


class LinearStochastic(eqx.Module):
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

    Parameters $w_e$, $\theta_e$, $w_s$, and $w_l$ constitute a random vector following a multivariate log-normal 
    distribution with tunable parameters $\mu$ and $\Sigma$.
    $\mu$ = [1.5%, 20°, 1, 0.5%] and $\Sigma$ = diag([0.1, 10., 1., 0.01]) by default.

    Attributes
    ----------
    mu: Float[Array, "4"]
        The mean vector of the model coefficients, **in the log-space**.
    cholesky_diag: Float[Array, "4"]
        The diagonal of the Cholesky decomposition corresponding to the covariance matrix of the model coefficients, 
        **in the log-space**.
    cholesky_tril: Float[Array, "6"]
        The lower triangle of the Cholesky decomposition corresponding to the covariance matrix of the model 
        coefficients, **in the log-space**.

    Methods
    -------
    __call__(t, y, args)
        Computes the drift velocity as a linear combination of surface currents (geostrophic), Ekman currents, 
        Stokes drift, and leeway velocity.
    """

    mu: Float[Array, "4"] = eqx.field(converter=lambda x: jnp.asarray(x))
    cholesky_diag: Float[Array, "4"] = eqx.field(converter=lambda x: jnp.asarray(x))
    cholesky_tril: Float[Array, "6"] = eqx.field(converter=lambda x: jnp.asarray(x))

    def get_cholesky(self) -> Float[Array, "4 4"]:
        """
        Return the Cholesky matrix from its (guaranteed positive) diagonal and lower triangle.

        Returns
        -------
        Float[Array, "4 4"]
            The Cholesky matrix from its (guaranteed positive) diagonal and lower triangle.
        """
        var = jax.nn.softplus(self.cholesky_diag) + 1e-6
        l = jnp.diag(var)
        l = l.at[jnp.tril_indices_from(l, -1)].set(self.cholesky_tril)
        return l

    def get_sigma(self) -> Float[Array, "4 4"]:
        l = self.get_cholesky()
        sigma = l @ l.T
        return sigma

    def get_coefficients(self) -> tuple[Float[Array, "4"], Float[Array, "4 4"]]:
        """
        Returns the model coefficients in the physical space.

        Returns
        -------
        tuple[Float[Array, "4"], Float[Array, "4 4"]]
            The model coefficients in the physical space.
        """
        sigma = self.get_sigma()
        mu_phy = jnp.exp(self.mu + 0.5 * jnp.diag(sigma))
        sigma_phy = jnp.outer(mu_phy, mu_phy) * (jnp.exp(sigma) - 1)
        
        return mu_phy, sigma_phy

    def get_gradients( self) -> tuple[Float[Array, "4"], Float[Array, "4"], Float[Array, "6"]]:
        """
        Returns the model coefficients gradients.

        Returns
        -------
        tuple[Float[Array, "4"], Float[Array, "4"], Float[Array, "6"]]
            The model coefficients gradients.
        """
        return self.mu, self.cholesky_diag, self.cholesky_tril

    def log_normal_transform(self, z: Float[Array, "4"]) -> Float[Array, "4"]:
        """
        Transforms a vector of independant random variables sampled from a standard normal distribution into a random 
        vector sampled from a multivariate log-normal distribution using the reparametrization trick.

        Parameters
        ----------
        z : Float[Array, "4"]
            A vector of independant random variables sampled from a standard normal distribution.

        Returns
        -------
        Float[Array, "4"]
            The corresponding random vector following a multivariate log-normal distribution.
        """
        l = self.get_cholesky()
        y = self.mu + z @ l.T
        x = jnp.exp(y)
        return x

    def __call__(
        self, t: Real[Array, ""], y: Float[Array, "2"], args: tuple[Gridded, Gridded, Gridded, Float[Array, "T 4"]]
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
            k = (2 * jnp.pi / tp)**2 / g  # wavenumber
            drogue_depth = 15
            stokes_drift = vu0 * (1 - jnp.exp(-2 * k * drogue_depth)) / (2 * k) / drogue_depth
            return stokes_scale * sanitize(stokes_drift)

        def leeway():
            return leeway_scale * sanitize(uw_vu)

        latitude, longitude = y
        longitude = longitude_in_180_180_degrees(longitude)  # ensure longitude is in [-180, 180] degrees

        ug_field, uw_field, uh_field, (zs, t0, dt) = args
        
        ug_vu = interp(ug_field, ("v", "u"), t, latitude, longitude)
        uw_vu = interp(uw_field, ("v", "u"), t, latitude, longitude)
        uh_vut = interp(uh_field, ("v", "u", "t"), t, latitude, longitude)

        z = zs[jnp.array((t - t0) // dt, int)]  # avoid sampling at each integration time step (dirty)
        ekman_scale, ekman_rotation, stokes_scale, leeway_scale = self.log_normal_transform(z)

        vu = geostrophy() + ekman() + stokes() + leeway()

        if ug_field.is_spherical_mesh and not ug_field.use_degrees:
            vu = meters_to_degrees(vu, latitude=latitude)

        return vu

    @classmethod
    def from_physical_space(
        cls, 
        mu_phy: Float[Array, "4"] | None = None, 
        sigma_phy: Float[Array, "4 4"] | None = None
    ):
        """
        Returns a model initialized with the mean and covariance matrix of the model coefficients in the log-space, 
        given their physical space counterparts.

        Parameters
        ----------
        mu_phy : Float[Array, "4"], optional
            The mean vector of the model coefficients in the physical space, by default `[1.5%, 20°, 1, 0.5%]`.
        sigma_phy : Float[Array, "4 4"], optional
            The covariance matrix of the model coefficients in the physical space, 
            by default `diag([1.5%, 20°, 1, 0.5%] / 10)`.

        Returns
        -------
        LinearStochastic
            A LinearStochastic model initialized with the given parameters.
        """
        if mu_phy is None:
            mu_phy = jnp.asarray([1.5 / 100, jnp.deg2rad(20.), 1., .5 / 100])
        if sigma_phy is None:
            sigma_phy = jnp.diag(mu_phy / 10)  # purely arbitrary
        
        sigma = jnp.log(sigma_phy / jnp.outer(mu_phy, mu_phy) + 1)
        var = jnp.diag(sigma)
        mu = jnp.log(mu_phy) - .5 * var
        cholesky_diag = softplus_inverse(jnp.sqrt(var))  # because sigma is diagonal

        return cls(mu=mu, cholesky_diag=cholesky_diag, cholesky_tril=jnp.zeros(6))
