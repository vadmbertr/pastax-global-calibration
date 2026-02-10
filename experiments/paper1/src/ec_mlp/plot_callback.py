
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import equinox as eqx
import jax.numpy as jnp
import lightning.pytorch as L
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.commons.callbacks import OnTrainEndCallback


def plot_fn(data_path: str, trainer: L.Trainer, trainer_module: L.LightningModule):
    drift_model = type(trainer_module).load_from_checkpoint(
        "best_model.ckpt", drift_model=trainer_module.drift_model
    ).drift_model

    lats = np.arange(-70, 71, 1)
    lons = np.arange(-180, 181, 1)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")

    months = np.arange(1, 13)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    inputs_3d = np.zeros((len(months), len(lats), len(lons), 3))
    inputs_3d[..., 0] = np.repeat(months[:, np.newaxis, np.newaxis], len(lats), axis=1).repeat(len(lons), axis=2)
    inputs_3d[..., 1] = np.repeat(grid_lat[np.newaxis, ..., :], len(months), axis=0)
    inputs_3d[..., 2] = np.repeat(grid_lon[np.newaxis, ..., :], len(months), axis=0)
    inputs_1d = inputs_3d.reshape(-1, 3)

    physical_parameters, mdn_parameters = eqx.filter_vmap(
        lambda arr: drift_model.get_physical_and_mdn_parameters(
            arr[0], arr[1], arr[2], to_physical_space=True, in_degrees=True
        )
    )(jnp.asarray(inputs_1d))

    beta_e, theta_e, beta_w = physical_parameters
    pi_k, mu_k, sigma_k, rho_k = mdn_parameters

    u_eps_mean, v_eps_mean = eqx.filter_vmap(drift_model.get_mean_residual_velocity_from_mdn_parameters)(pi_k, mu_k)
    dx_eps_mean = u_eps_mean * drift_model.delta_t
    dy_eps_mean = v_eps_mean * drift_model.delta_t

    u_eps_mode, v_eps_mode = eqx.filter_vmap(drift_model.get_mode_residual_velocity_from_mdn_parameters)(pi_k, mu_k)
    dx_eps_mode = u_eps_mode * drift_model.delta_t
    dy_eps_mode = v_eps_mode * drift_model.delta_t

    K_crossflow = eqx.filter_vmap(drift_model.get_crossflow_diffusivity_from_mdn_parameters)(
        pi_k, mu_k, sigma_k, rho_k
    )

    beta_e_map = beta_e.reshape(len(months), len(lats), len(lons))
    theta_e_map = theta_e.reshape(len(months), len(lats), len(lons))
    beta_w_map = beta_w.reshape(len(months), len(lats), len(lons))

    u_eps_mean_map = u_eps_mean.reshape(len(months), len(lats), len(lons))
    v_eps_mean_map = v_eps_mean.reshape(len(months), len(lats), len(lons))
    u_eps_mode_map = u_eps_mode.reshape(len(months), len(lats), len(lons))
    v_eps_mode_map = v_eps_mode.reshape(len(months), len(lats), len(lons))
    dx_eps_mean_map = dx_eps_mean.reshape(len(months), len(lats), len(lons))
    dy_eps_mean_map = dy_eps_mean.reshape(len(months), len(lats), len(lons))
    dx_eps_mode_map = dx_eps_mode.reshape(len(months), len(lats), len(lons))
    dy_eps_mode_map = dy_eps_mode.reshape(len(months), len(lats), len(lons))
    K_crossflow_map = K_crossflow.reshape(len(months), len(lats), len(lons))

    # get land-sea mask
    ds = xr.open_zarr(f"{data_path}/sea_land_mask.zarr")
    mask = ds["is_sea"]
    mask_int = mask.astype(int)
    mask_coarse = mask_int.coarsen(latitude=4, longitude=4, boundary="trim").mean()
    mask_1deg = (mask_coarse.interp(latitude=lats, longitude=lons, method="nearest") > 0.5).values

    out_dir = trainer.log_dir
    writer = None
    for logger in trainer.loggers:
        if isinstance(logger, L.loggers.TensorBoardLogger):
            writer = logger.experiment

    def make_plots(name, arr, cmap, vmin, vmax, unit=""):
        def make_map_plot(da_, fig_title, fig_name):
            fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

            da_.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, color="grey")
            ax.set_title(f"{fig_title}")

            fig.tight_layout()

            save_fig(fig, fig_name)

            plt.close(fig)

        def save_fig(fig, fig_name):
            os.makedirs(os.path.join(out_dir, name), exist_ok=True)
            fig.savefig(os.path.join(out_dir, name, f"{fig_name}.png"))
            if writer:
                writer.add_figure(f"{name}/{fig_name}", fig, trainer.global_step)

        da = xr.DataArray(
            arr,
            coords={
                "month": months,
                "latitude": lats,
                "longitude": lons
            },
            dims=["month", "latitude", "longitude"],
            attrs={"long_name": name, "units": unit}
        )
        da = da.where(mask_1deg)

        # per month
        for month in months:
            month_name = month_names[month - 1]
            make_map_plot(da.isel(month=month - 1), f"{name} - {month_name}", f"{name}_{month_name}")

        # annual mean
        make_map_plot(da.mean(dim="month"), f"{name} - Annual Mean", f"{name}_annual_mean")

        # spatial mean, split North and South emisphere
        da_north = da.sel(latitude=slice(0, 70))
        da_south = da.sel(latitude=slice(-70, 0))

        if name == "theta_e":
            da_north = np.abs(da_north)  # for visualization purposes, take absolute value of theta_e

        fig, ax = plt.subplots()

        da_north.weighted(np.cos(np.deg2rad(da_north.latitude))).mean(
            dim=["latitude", "longitude"]
        ).plot(ax=ax, label=f"Northern Hemisphere", linestyle="-", marker="o")
        da_south.weighted(np.cos(np.deg2rad(da_south.latitude))).mean(
            dim=["latitude", "longitude"]
        ).plot(ax=ax, label=f"Southern Hemisphere", linestyle="--", marker="x")
        ax.legend()
        ax.set_title(f"{name} - Spatial Mean")

        fig.tight_layout()

        save_fig(fig, f"{name}_spatial_mean")

        plt.close(fig)


    def extended_center_cmap(base_cmap, vmin, vcenter_low, vcenter_high, vmax, n=256):
        lower = (vcenter_low - vmin) / (vmax - vmin)
        upper = (vcenter_high - vmin) / (vmax - vmin)

        colors_low  = base_cmap(np.linspace(0.0, 0.5, int(n * lower)))
        colors_mid  = np.ones((int(n * (upper - lower)), 4))  # pure white
        colors_high = base_cmap(np.linspace(0.5, 1.0, n - len(colors_low) - len(colors_mid)))

        colors = np.vstack((colors_low, colors_mid, colors_high))
        return ListedColormap(colors)

    theta_e_cmap = extended_center_cmap(base_cmap=cmo.balance, vmin=-135, vcenter_low=-45, vcenter_high=45, vmax=135)

    make_plots("beta_e", beta_e_map, cmap=cmo.amp, vmin=0, vmax=None, unit="$m^2s/kg$")
    make_plots("theta_e", theta_e_map, cmap=theta_e_cmap, vmin=-135, vmax=135, unit="°")
    make_plots("beta_w", beta_w_map, cmap=cmo.amp, vmin=0, vmax=None, unit="%")
    make_plots("u_eps_mean", u_eps_mean_map, cmap=cmo.diff, vmin=-0.25, vmax=0.25, unit="m/s")
    make_plots("v_eps_mean", v_eps_mean_map, cmap=cmo.diff, vmin=-0.25, vmax=0.25, unit="m/s")
    make_plots("u_eps_mode", u_eps_mode_map, cmap=cmo.diff, vmin=-0.25, vmax=0.25, unit="m/s")
    make_plots("v_eps_mode", v_eps_mode_map, cmap=cmo.diff, vmin=-0.25, vmax=0.25, unit="m/s")
    make_plots("dx_eps_mean", dx_eps_mean_map / 1000, cmap=cmo.delta, vmin=-1.5, vmax=1.5, unit="km")
    make_plots("dy_eps_mean", dy_eps_mean_map / 1000, cmap=cmo.delta, vmin=-1.5, vmax=1.5, unit="km")
    make_plots("dx_eps_mode", dx_eps_mode_map / 1000, cmap=cmo.delta, vmin=-1.5, vmax=1.5, unit="km")
    make_plots("dy_eps_mode", dy_eps_mode_map / 1000, cmap=cmo.delta, vmin=-1.5, vmax=1.5, unit="km")
    make_plots("K_crossflow", K_crossflow_map, cmap=cmo.amp, vmin=0, vmax=None, unit="$m^2/s$")


class PlotCallback(OnTrainEndCallback):
    def __init__(self, data_path: str):
        super().__init__(lambda *args: plot_fn(data_path, *args))
