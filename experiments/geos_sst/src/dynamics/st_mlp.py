from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, Int, Real
from kymatio.jax import Scattering2D
import lineax as lx

from pastax.gridded import Gridded
from pastax.utils import longitude_in_180_180_degrees, meters_to_degrees


class StMlp(eqx.Module):

    scattering_2d: Scattering2D = eqx.field(static=True)
    mlp_encoder: eqx.nn.Sequential
    J: int = eqx.field(static=True)
    L: int = eqx.field(static=True)
    M: int = eqx.field(static=True)
    use_iso_indexes_only: bool = eqx.field(static=True)

    def __call__(
        self, t: Real[Array, ""], y: Float[Array, "2"], args: tuple[Gridded, Gridded]
    ) -> Float[Array, "2"]:
        def sanitize(arr: Float[Array, "..."], replacement: Float[Array, ""] = 0.0) -> Float[Array, "..."]:
            return jnp.nan_to_num(arr, nan=replacement, posinf=replacement, neginf=replacement)
        
        latitude, longitude = y
        longitude = longitude_in_180_180_degrees(longitude)  # ensure longitude is in [-180, 180] degrees

        duacs_ds, mur_ds = args
        
        vu_geos = self._geostrophic_part(latitude, longitude, t, duacs_ds)
        mu, sigma = self._learned_part(latitude, longitude, t, mur_ds)

        vu_geos = sanitize(vu_geos)
        mu = sanitize(mu)
        sigma = sanitize(sigma, replacement=1.0)

        vu_deter = vu_geos + mu
        vu_stoch = sigma

        if duacs_ds.is_spherical_mesh and not duacs_ds.use_degrees:
            vu_deter = meters_to_degrees(vu_deter, latitude=latitude)
            vu_stoch = meters_to_degrees(vu_stoch, latitude=latitude)

        return lx.PyTreeLinearOperator((vu_deter, vu_stoch), jax.ShapeDtypeStruct((2,), float))

    @classmethod
    def _geostrophic_part(
        cls, latitude: Float[Array, ""], longitude: Float[Array, ""], t: Real[Array, ""], duacs_ds: Gridded
    ) -> Float[Array, "2"]:
        def interp(field, variables, t, lat, lon):
            uv_dict = field.interp(*variables, time=t, latitude=lat, longitude=lon)
            return jnp.asarray([uv_dict[k] for k in variables])
        
        vu_geos = interp(duacs_ds, ("v", "u"), t, latitude, longitude)
        return jnp.nan_to_num(vu_geos, nan=0.0, posinf=0.0, neginf=0.0)

    def _learned_part(
        self, latitude: Float[Array, ""], longitude: Float[Array, ""], t: Real[Array, ""], mur_ds: Gridded
    ) -> tuple[Float[Array, "2"], Float[Array, "2 2"]]:
        def jacobi_filling(field, num_iters=100):
            def initial_guess(field_with_nans):
                row_mean = jnp.nanmean(field_with_nans, axis=1, keepdims=True)
                col_mean = jnp.nanmean(field_with_nans, axis=0, keepdims=True)

                row_mean = jnp.broadcast_to(row_mean, field_with_nans.shape)
                col_mean = jnp.broadcast_to(col_mean, field_with_nans.shape)

                row_col_mean = (row_mean + col_mean) / 2

                filled_mean = jnp.where(nan_mask, row_col_mean, field_with_nans)

                return filled_mean

            def body_fn(filled_old):
                padded = jnp.pad(filled_old, 1, mode="reflect")

                up    = padded[:-2, 1:-1]
                down  = padded[2:, 1:-1]
                left  = padded[1:-1, :-2]
                right = padded[1:-1, 2:]
                neighbor_avg = (up + down + left + right) / 4

                filled_new = jnp.where(nan_mask, neighbor_avg, filled_old)

                return filled_new

            def scan_fn(carry, _):
                filled_old = carry
                filled_new = body_fn(filled_old)
                return filled_new, None

            nan_mask = jnp.isnan(field)

            filled_final, _ = jax.lax.cond(
                jnp.sum(nan_mask) == 0,
                lambda _: (field, None),
                lambda _: jax.lax.scan(scan_fn, initial_guess(field), None, length=num_iters),
                None
            )

            return filled_final

        def compute_isotropic_indexes(S1: Float[Array, "..."], S2: Float[Array, "..."]) -> Float[Array, "..."]:
            def get_S2_indice(j1, l1, j2, l2):
                offset = sum((self.J - (j1p + 1)) * self.L * self.L for j1p in range(j1))
                idx_in_block = l1 * (self.J - (j1 + 1)) * self.L + (j2 - (j1 + 1)) * self.L + l2
                return offset + idx_in_block

            isotropic_indexes = jnp.zeros(self.J * (self.J - 1))
            glob_i = 0

            for j1 in range(self.J):
                for j2 in range(j1 + 1, self.J):
                    acc = 0
                    for l1 in range(self.L):
                        for l2 in range(self.L):
                            acc += (S2[get_S2_indice(j1, l1, j2, l2)] / S1[j1 * self.L + l1])
                    isotropic_indexes = isotropic_indexes.at[glob_i].set(acc / (self.L ** 2))
                    glob_i += 1

            for j1 in range(self.J):
                for j2 in range(j1 + 1, self.J):
                    acc = 0
                    for l1 in range(self.L):
                        l2 = (l1 + self.L // 2) % self.L
                        acc += (S2[get_S2_indice(j1, l1, j2, l1)] / S2[get_S2_indice(j1, l1, j2, l2)])
                    isotropic_indexes = isotropic_indexes.at[glob_i].set(acc / self.L)
                    glob_i += 1

            return isotropic_indexes

        mur_ds = mur_ds.neighborhood("T", time=t, latitude=latitude, longitude=longitude, t_width=1, x_width=self.M)
        sst = mur_ds.fields["T"].values[0, ...]

        sst = jacobi_filling(sst)
        st_coeffs = self.scattering_2d(sst)

        if self.use_iso_indexes_only:
            st_coeffs = compute_isotropic_indexes(st_coeffs[1:(self.J * self.L + 1)], st_coeffs[(self.J * self.L + 1):])
        else:
            st_coeffs = jnp.mean(st_coeffs[..., 1:-1], axis=(-1, -2))

        mu_y, mu_x, sigma_y, sigma_x, sigma_yx = self.mlp_encoder(st_coeffs)

        mu = jnp.asarray((mu_x, mu_y))
        sigma = jnp.diag(jnp.exp(jnp.asarray((sigma_y, sigma_x))))
        sigma = sigma.at[0, 1].set(sigma_yx)
        sigma = sigma.at[1, 0].set(sigma_yx)

        return mu, sigma

    @classmethod
    def from_hyperparameters(
        cls, 
        J: int = 3,
        L: int = 4,
        M: int = 50,
        H1: Int[Array, ""] = 24,
        H2: Int[Array, ""] = 12,
        use_iso_indexes_only: bool = False
    ):
        key = jrandom.key(0)
        key1, key2, key3 = jrandom.split(key, 3)

        scattering_2d = Scattering2D(J=J, shape=(M, M), L=L)
        
        if use_iso_indexes_only:
            in_features = J * (J - 1)
        else:
            in_features = 1 + J * L + J * (J - 1) * L ** 2 / 2
        in_features = int(in_features)
        out_features = 5

        mlp_encoder = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(in_features),
                eqx.nn.Linear(in_features, H1, key=key1),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Linear(H1, H2, key=key2),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Linear(H2, out_features, key=key3)
            ]
        )
        
        return cls(
            scattering_2d=scattering_2d, mlp_encoder=mlp_encoder, 
            J=J, L=L, M=M, 
            use_iso_indexes_only=use_iso_indexes_only
        )
