from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, Real
import lineax as lx

from pastax.gridded import Gridded
from pastax.utils import longitude_in_180_180_degrees, meters_to_degrees


class CNN(eqx.Module):

    mlp_encoder: eqx.nn.Sequential
    M: int = eqx.field(static=True)

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
        
        return interp(duacs_ds, ("v", "u"), t, latitude, longitude)

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

        mur_ds = mur_ds.neighborhood("T", time=t, latitude=latitude, longitude=longitude, t_width=1, x_width=self.M)
        sst = mur_ds.fields["T"].values[0, ...]

        sst = jacobi_filling(sst)

        mu_y, mu_x, sigma_y, sigma_x, sigma_yx = self.mlp_encoder(sst)

        mu = jnp.asarray((mu_x, mu_y))
        sigma = jnp.diag(jnp.exp(jnp.asarray((sigma_y, sigma_x))))
        sigma = sigma.at[0, 1].set(sigma_yx)
        sigma = sigma.at[1, 0].set(sigma_yx)

        return mu, sigma

    @classmethod
    def from_hyperparameters(
        cls, 
        M: int = 50,
        layers: list[dict[str, int | Sequence[int]]] = [
            {"out_channels": 16, "kernel_size": (5, 5), "stride": 2, "padding": 2},
            {"out_channels": 32, "kernel_size": (3, 3), "stride": 2, "padding": 1},
            {"out_channels": 64, "kernel_size": (3, 3), "stride": 2, "padding": 1}
        ]
    ):
        key = jrandom.key(0)
        keys = jrandom.split(key, len(layers))

        cnn_layers = [eqx.nn.LayerNorm((M, M)),]
        in_channels = 1
        for key, layer in zip(keys, layers):
            cnn_layers.append(eqx.Conv2d(in_channels=in_channels, key=key, **layer))
            cnn_layers.append(eqx.nn.Lambda(jax.nn.gelu))
            in_channels = layers["out_channels"]

        cnn_layers.append(eqx.nn.AdaptiveAvgPool2d(64))
        cnn_layers.append(eqx.nn.Linear(64, 5))

        cnn_encoder = eqx.nn.Sequential(cnn_layers)

        return cls(cnn_encoder=cnn_encoder, M=M)
