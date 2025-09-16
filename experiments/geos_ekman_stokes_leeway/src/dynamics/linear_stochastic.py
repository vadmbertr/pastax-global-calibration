from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Real

from pastax.gridded import Gridded
from pastax.utils import longitude_in_180_180_degrees, meters_to_degrees


def sanitize(arr: Real[Array, "..."]) -> Real[Array, "..."]:
    return jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


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
    $\mathbf{u}_h = \mathbf{u}_{h_0} \frac{1 - \exp{-2k_m |z_d|}}{2k |z_d|}$, where k is the wave number 
    and $z_d$ is the depth of the drogue (-15m).

    Parameters $w_e$, $\theta_e$, $w_s$, and $w_l$ constitute a random vector following a multivariate log-normal 
    distribution with tunable parameters $\mu$ and $\Sigma$.
    $\mu$ = [1.5%, 45°, 1, 0.5%] and $\Sigma$ = diag($\mu$ / 10) by default.

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
    deterministic_mode: bool = eqx.field(static=True)
    depth_integrated_stokes: bool = eqx.field(static=True)
    effective_wavenumber: bool = eqx.field(static=True)
    include_leeway: bool = eqx.field(static=True)

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
        return sanitize(l)

    def get_mu(self) -> Float[Array, "4"]:
        return sanitize(self.mu)

    def get_sigma(self) -> Float[Array, "4 4"]:
        l = self.get_cholesky()
        sigma = l @ l.T
        return sanitize(sigma)

    def get_coefficients(self) -> tuple[Float[Array, "4"], Float[Array, "4 4"]]:
        """
        Returns the model coefficients in the physical space.

        Returns
        -------
        tuple[Float[Array, "4"], Float[Array, "4 4"]]
            The model coefficients in the physical space.
        """
        sigma = self.get_sigma()
        mu_phy = jnp.exp(self.get_mu() + 0.5 * jnp.diag(sigma))
        mu_phy = sanitize(mu_phy)
        sigma_phy = jnp.outer(mu_phy, mu_phy) * (jnp.exp(sigma) - 1)
        sigma_phy = sanitize(sigma_phy)

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
        y = self.get_mu() + z @ l.T
        x = jnp.exp(y)
        return x

    def log_normal_joint_mode(self) -> Float[Array, "4"]:
        """
        
        """
        mu = self.get_mu()
        sigma = self.get_sigma()
        joint_mode = jnp.exp(mu - jnp.sum(sigma, axis=1))  # equiv. to jnp.exp(mu - sigma @ jnp.ones(mu.shape))
        return joint_mode

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

        if not self.deterministic_mode:
            ug_field, uw_field, uh_field, (zs, t0, dt) = args
        else:
            ug_field, uw_field, uh_field, _ = args

        ug_vu = interp(ug_field, ("v", "u"), t, latitude, longitude)
        uw_vu = interp(uw_field, ("v", "u"), t, latitude, longitude)
        uh_vuth = interp(uh_field, ("v", "u", "t", "h"), t, latitude, longitude)

        if not self.deterministic_mode:
            z = zs[jnp.array((t - t0) // dt, int)]  # avoid sampling at each integration time step (dirty)
            ekman_scale, ekman_rotation, stokes_scale, leeway_scale = self.log_normal_transform(z)
        else:
            ekman_scale, ekman_rotation, stokes_scale, leeway_scale = self.log_normal_joint_mode()

        vu = geostrophy() + ekman() + stokes() + leeway()

        if ug_field.is_spherical_mesh and not ug_field.use_degrees:
            vu = meters_to_degrees(vu, latitude=latitude)

        return vu

    @classmethod
    def from_physical_space(
        cls, 
        mu_phy: Float[Array, "4"] | None = None, 
        sigma_phy: Float[Array, "4 4"] | None = None,
        deterministic_mode: bool = False,
        depth_integrated_stokes: bool = True,
        effective_wavenumber: bool = True,
        include_leeway: bool = True
    ):
        """
        Returns a model initialized with the mean and covariance matrix of the model coefficients in the log-space, 
        given their physical space counterparts.

        Parameters
        ----------
        mu_phy : Float[Array, "4"], optional
            The mean vector of the model coefficients in the physical space, by default `[1.5%, 45°, 1, 0.5%]`.
        sigma_phy : Float[Array, "4 4"], optional
            The covariance matrix of the model coefficients in the physical space, 
            by default `diag(mu_phy / 10)`.

        Returns
        -------
        LinearStochastic
            A LinearStochastic model initialized with the given parameters.
        """
        if mu_phy is None:
            mu_phy = jnp.asarray([1.5 / 100, jnp.deg2rad(45.), 1., .5 / 100])
        if sigma_phy is None:
            sigma_phy = jnp.diag(mu_phy / 10)  # purely arbitrary
        
        sigma = jnp.log(sigma_phy / jnp.outer(mu_phy, mu_phy) + 1)
        var = jnp.diag(sigma)
        mu = jnp.log(mu_phy) - .5 * var
        cholesky_diag = softplus_inverse(jnp.sqrt(var))  # because sigma is diagonal

        return cls(
            mu=mu,
            cholesky_diag=cholesky_diag,
            cholesky_tril=jnp.zeros(6),
            deterministic_mode=deterministic_mode,
            depth_integrated_stokes=depth_integrated_stokes,
            effective_wavenumber=effective_wavenumber,
            include_leeway=include_leeway
        )
