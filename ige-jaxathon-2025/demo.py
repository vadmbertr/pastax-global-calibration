import time

import clouddrift as cd
import dask
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import optax
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import xarray as xr

from pastax.dynamics import StochasticSmagorinskyDiffusion
from pastax.gridded import Gridded
from pastax.simulator import StochasticSimulator
from pastax.trajectory import Trajectory

jax.config.update("jax_enable_x64", True)

DATA_ROOT = "/summer/meom"

# trajectory related
N_DAYS = 5
INTEGRATION_DT = 30 * 60  # in seconds
N_STEPS = N_DAYS * 24 * 60 * 60 // INTEGRATION_DT
ENSEMBLE_SIZE = 50

# dataloading related
DASK_N_WORKERS = 8
DL_N_WORKERS = 24
BATCH_SIZE = 128
PREFETCH_FACTOR = 3

# optimization related
N_EPOCHS = 10
LEARNING_RATE = 1e-3

drifter_ds = xr.open_zarr(f"{DATA_ROOT}/workdir/bertrava/noaa-oar-hourly-gdp-pds.zarr")
drifter_ds

def chunk_trajectories(
    ds: xr.Dataset, n_days: int = N_DAYS, dt: np.timedelta64 = np.timedelta64(1, "h"), to_ragged: bool = False
) -> xr.Dataset:
    def ragged_chunk(arr: xr.DataArray | np.ndarray, is_metadata: bool = False) -> np.ndarray:
        arr = cd.ragged.apply_ragged(cd.ragged.chunk, arr, row_size, chunk_size)  # noqa
        if is_metadata:
            arr = arr[:, 0]
        if to_ragged:
            arr = arr.ravel()
        return arr

    if dt is None:
        dt = (ds.isel(traj=0).time[1] - ds.isel(traj=0).time[0])

    row_size = cd.ragged.segment(ds.time, dt, ds.rowsize)  # if holes, divide into segments
    chunk_size = int(n_days / (dt / np.timedelta64(1, "D"))) + 1

    # chunk along `obs` dimension (data)
    data = dict(
        [(d, ragged_chunk(ds[d])) for d in ["time", "lat", "lon"]]
    )

    # chunk along `traj` dimension (metadata)
    metadata = {"id": ragged_chunk(np.repeat(ds["id"], ds.rowsize), is_metadata=True)}
    metadata["rowsize"] = np.full(metadata["id"].size, chunk_size)  # noqa - after chunking the rowsize is constant

    # create xr.Dataset
    attrs_global = ds.attrs

    coord_dims = {}
    attrs_variables = {}
    for var in ds.coords.keys():
        var = str(var)
        coord_dims[var] = str(ds[var].dims[-1])
        attrs_variables[var] = ds[var].attrs

    for var in data.keys():
        attrs_variables[var] = ds[var].attrs

    for var in metadata.keys():
        attrs_variables[var] = ds[var].attrs

    metadata["drifter_id"] = metadata["id"]  # noqa
    del metadata["id"]
    attrs_variables["drifter_id"] = attrs_variables["id"]
    attrs_variables["id"] = {}

    if to_ragged:
        coords = {"id": np.arange(metadata["drifter_id"].size), "time": data.pop("time")}
        ragged_array = cd.RaggedArray(
            coords, metadata, data, attrs_global, attrs_variables, {"traj": "rows", "obs": "obs"}, coord_dims
        )
        ds = ragged_array.to_xarray()
    else:
        coords = {"id": np.arange(metadata["drifter_id"].size)}
        
        xr_coords = {}
        for var in coords.keys():
            xr_coords[var] = (
                [coord_dims[var]],
                coords[var],
                attrs_variables[var],
            )

        xr_data = {}
        for var in metadata.keys():
            xr_data[var] = (
                ["traj"],
                metadata[var],
                attrs_variables[var],
            )

        for var in data.keys():
            xr_data[var] = (
                ["traj", "obs"],
                data[var],
                attrs_variables[var],
            )

        ds = xr.Dataset(coords=xr_coords, data_vars=xr_data, attrs=attrs_global)

    return ds

traj_ds = chunk_trajectories(drifter_ds)
traj_ds

ssc_ds = xr.open_zarr(f"{DATA_ROOT}/workdir/bertrava/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D.zarr")
ssc_ds

class Dataset(TorchDataset):
    def __init__(self, traj_ds: xr.Dataset, ssc_ds: xr.Dataset, ssc_ds_periodic: bool = True):
        self.traj_ds = traj_ds
        self.ssc_ds = ssc_ds
        self.ssc_ds_periodic = ssc_ds_periodic

        max_travel_distance = .5  # in ° / day ; inferred from data
        traj_t0_t1 = traj_ds.time.isel(traj=0)[np.asarray([0, -1])]
        n_days = ((traj_t0_t1[-1] - traj_t0_t1[0]) / np.timedelta64(1, "D")).astype(int).item()
        self.max_travel_distance = max_travel_distance * n_days

        self.ssc_nt = ssc_ds.time.size
        self.ssc_mint = ssc_ds.time.min()
        self.ssc_dt = ssc_ds.time[1] - ssc_ds.time[0]
        self.ssc_t_di = np.ceil(np.timedelta64(n_days, "D") / self.ssc_dt)
        self.ssc_nlat = ssc_ds.latitude.size
        self.ssc_minlat = ssc_ds.latitude.min()
        self.ssc_dlat = ssc_ds.latitude[1] - ssc_ds.latitude[0]  # regular grid
        self.ssc_lat_di = np.ceil(self.max_travel_distance / self.ssc_dlat)
        self.ssc_nlon = ssc_ds.longitude.size
        self.ssc_minlon = ssc_ds.longitude.min()
        self.ssc_dlon = ssc_ds.longitude[1] - ssc_ds.longitude[0]  # regular grid
        self.ssc_lon_di = np.ceil(self.max_travel_distance / self.ssc_dlon)

    def __len__(self):
        return self.traj_ds.traj.size

    def __getitem__(self, idx: int):
        traj_arrays = self.__get_traj_arrays(idx)
        ssc_arrays = self.__get_ssc_arrays(*traj_arrays[:3])
        
        return traj_arrays, ssc_arrays  # we would like to return jax.Array
    
    def __get_traj_arrays(self, idx: int):
        traj_subset = self.traj_ds.isel(traj=idx)
        
        traj_lat = traj_subset.lat.values.ravel()
        traj_lon = traj_subset.lon.values.ravel()
        traj_time = traj_subset.time.values.ravel().astype("datetime64[s]").astype(int)  # in seconds
        traj_id = traj_subset.id.values.ravel()
        
        return traj_lat, traj_lon, traj_time, traj_id
    
    def __get_ssc_arrays(self, traj_lat, traj_lon, traj_time):
        def get_latlon_minmax(latlon0_i, latlon_di):
            min_i = (latlon0_i - latlon_di).astype(int).item()
            max_i = (latlon0_i + latlon_di).astype(int).item()
            return min_i, max_i

        def get_pads(min_i, max_i, n):
            padleft = max(0, -min_i)
            min_i = max(0, min_i)
            padright = max(0, max_i - (n - 1))
            max_i = min(n - 1, max_i)
            return (padleft, padright), (min_i, max_i)    

        t0 = traj_time[0].astype("datetime64[s]")
        lat0 = traj_lat[0]
        lon0 = traj_lon[0]

        t0_i = np.floor((t0 - self.ssc_mint) / self.ssc_dt)
        lat0_i = ((lat0 - self.ssc_minlat) / self.ssc_dlat).round()
        lon0_i = ((lon0 - self.ssc_minlon) / self.ssc_dlon).round()

        tmin_i = t0_i.astype(int).item()
        tmax_i = (t0_i + self.ssc_t_di).astype(int).item()
        latmin_i, latmax_i = get_latlon_minmax(lat0_i, self.ssc_lat_di)
        lonmin_i, lonmax_i = get_latlon_minmax(lon0_i, self.ssc_lon_di)

        (t_padleft, t_padright), (tmin_i, tmax_i) = get_pads(tmin_i, tmax_i, self.ssc_nt)
        (lat_padleft, lat_padright), (latmin_i, latmax_i) = get_pads(latmin_i, latmax_i, self.ssc_nlat)
        (lon_padleft, lon_padright), (lonmin_i, lonmax_i) = get_pads(lonmin_i, lonmax_i, self.ssc_nlon)

        ssc_patch = self.ssc_ds.isel(
            time=slice(tmin_i, tmax_i + 1),
            latitude=slice(latmin_i, latmax_i + 1), 
            longitude=slice(lonmin_i, lonmax_i + 1)
        )

        ssc_u = ssc_patch.ugos
        ssc_v = ssc_patch.vgos
        ssc_time = ssc_patch.time.astype("datetime64[s]").astype(int)  # in seconds
        ssc_lat = ssc_patch.latitude
        ssc_lon = ssc_patch.longitude

        if self.ssc_ds_periodic:  # periodic global domain
            if lon_padleft != 0:
                ssc_patch_left = self.ssc_ds.isel(
                    time=slice(tmin_i, tmax_i + 1),
                    latitude=slice(latmin_i, latmax_i + 1), 
                    longitude=slice(self.ssc_nlon - lon_padleft, self.ssc_nlon)  # right part goes to the left
                )
                ssc_u_left = ssc_patch_left.ugos
                ssc_v_left = ssc_patch_left.vgos
                ssc_lon_left = ssc_patch_left.longitude

                ssc_u = np.concat([ssc_u_left, ssc_u], axis=-1)
                ssc_v = np.concat([ssc_v_left, ssc_v], axis=-1)
                ssc_lon = np.concat([ssc_lon_left, ssc_lon])

                lon_padleft = 0

            if lon_padright != 0:
                ssc_patch_right = self.ssc_ds.isel(
                    time=slice(tmin_i, tmax_i + 1),
                    latitude=slice(latmin_i, latmax_i + 1), 
                    longitude=slice(0, lon_padright)  # left part goes to the right
                )
                ssc_u_right = ssc_patch_right.ugos
                ssc_v_right = ssc_patch_right.vgos
                ssc_lon_right = ssc_patch_right.longitude

                ssc_u = np.concat([ssc_u, ssc_u_right], axis=-1)
                ssc_v = np.concat([ssc_v, ssc_v_right], axis=-1)
                ssc_lon = np.concat([ssc_lon, ssc_lon_right])

                lon_padright = 0

        ssc_u = np.pad(
            ssc_u, ((t_padleft, t_padright), (lat_padleft, lat_padright), (lon_padleft, lon_padright)), mode="edge"
        )
        ssc_v = np.pad(
            ssc_v, ((t_padleft, t_padright), (lat_padleft, lat_padright), (lon_padleft, lon_padright)), mode="edge"
        )
        ssc_time = np.pad(ssc_time, (t_padleft, t_padright), mode="edge")
        ssc_lat = np.pad(ssc_lat, (lat_padleft, lat_padright), mode="edge")
        ssc_lon = np.pad(ssc_lon, (lon_padleft, lon_padright), mode="edge")
        
        return ssc_u, ssc_v, ssc_time, ssc_lat, ssc_lon

dataset = Dataset(traj_ds, ssc_ds)

xr_jax_dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=True,
    num_workers=DL_N_WORKERS, prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=True, 
    in_order=False, # multiprocessing_context="forkserver"
)

def torch_to_jax(array):
    return jnp.asarray(array.numpy())

@eqx.filter_jit  # this improves performances
def to_trajectories(traj_arrays):
    traj_lat, traj_lon, traj_time, traj_id = traj_arrays

    traj_latlon = jnp.stack((traj_lat, traj_lon), axis=-1)
    trajectories = eqx.filter_vmap(
        lambda _latlon, _time, _id: Trajectory.from_array(values=_latlon, times=_time, id=_id)
    )(
        traj_latlon, traj_time, traj_id
    )

    return trajectories


@eqx.filter_jit  # this improves performances
def to_gridded(ssc_arrays):
    ssc_u, ssc_v, ssc_time, ssc_lat, ssc_lon = ssc_arrays
    
    gridded = eqx.filter_vmap(Gridded.from_array)(
        {"u": ssc_u, "v": ssc_v}, ssc_time, ssc_lat, ssc_lon
    )

    return gridded

# configure Dask workers
dask.config.set(scheduler="threads", num_workers=DASK_N_WORKERS)

simulator = StochasticSimulator()
dynamics = StochasticSmagorinskyDiffusion.from_cs(cs=1e-1)

def forward(_dynamics, grid, x0, ts, key):
    dt0, saveat, stepsize_controller, adjoint, n_steps, brownian_motion = simulator.get_diffeqsolve_best_args(
        ts, INTEGRATION_DT, n_steps=N_STEPS, constant_step_size=True, save_at_steps=False, ad_mode="forward"
    )

    return simulator(
        dynamics=_dynamics, args=grid, x0=x0, ts=ts, 
        dt0=dt0, saveat=saveat, stepsize_controller=stepsize_controller, adjoint=adjoint, 
        max_steps=n_steps, key=key, brownian_motion=brownian_motion
    )

jitted_forward = eqx.filter_jit(forward)

def loss(_dynamics, _grid_batch, _reference_trajectory_batch, _key_batch):
    def pair_residual(traj1, traj2):
        residuals = traj1.liu_index(traj2).value
        residuals = jnp.where(jnp.isnan(residuals), 1, residuals)
        return residuals
    
    def _loss(grid, reference_trajectory, key):
        x0 = reference_trajectory.origin
        ts = reference_trajectory.times.value
        simulated_ensemble = forward(_dynamics, grid, x0, ts, key)
        residuals = simulated_ensemble.crps(reference_trajectory, pair_residual, is_metric_symmetric=False)
        return (residuals ** 2).sum().value
    
    loss_batch = eqx.filter_vmap(_loss)(_grid_batch, _reference_trajectory_batch, _key_batch)
    _loss = loss_batch.mean()

    return _loss, _loss  # for returning both grad and value in forward AD mode

jitted_loss = eqx.filter_jit(loss)
grad_val_loss = eqx.filter_jacfwd(loss, has_aux=True)
jitted_grad_val_loss = eqx.filter_jit(grad_val_loss)

@eqx.filter_jit
def make_step(_dynamics, _opt_state, _grid_batch, _reference_trajectory_batch, _key_batch, _optim):
    grad, loss_val = grad_val_loss(_dynamics, _grid_batch, _reference_trajectory_batch, _key_batch)
    updates, _opt_state = _optim.update(grad, _opt_state)
    _dynamics = eqx.apply_updates(_dynamics, updates)
    return _dynamics, _opt_state, loss_val

key = jrd.key(0)
optim = optax.chain(optax.zero_nans(), optax.adam(LEARNING_RATE))
opt_state = optim.init(dynamics)

losses = []
cs = []

t0 = time.time()
n_samples = 0
data_loading_time = 0
compute_time = 0
for epoch in range(1):
    key = jrd.split(key, 1)[0]
    t1 = time.time()
    for step, (reference_trajectory_batch, grid_batch) in enumerate(xr_jax_dataloader):
        reference_trajectory_batch = [torch_to_jax(arr) for arr in reference_trajectory_batch]
        grid_batch = [torch_to_jax(arr) for arr in grid_batch]

        batch_size = reference_trajectory_batch[0].shape[0]

        batch_key, key = jrd.split(key, 2)
        key_batch = jrd.split(batch_key, batch_size)
        
        reference_trajectory_batch = to_trajectories(reference_trajectory_batch)
        grid_batch = to_gridded(grid_batch)
        
        data_loading_time += time.time() - t1
        t1 = time.time()

        dynamics, opt_state, loss_val = make_step(
            dynamics, opt_state, grid_batch, reference_trajectory_batch, key_batch, optim
        )
        losses.append(loss_val.item())
        cs.append(dynamics.cs.item())

        n_samples += batch_size
        compute_time += time.time() - t1
        t1 = time.time()

total_time = time.time() - t0

print(f"Calibrate using {n_samples} samples in {total_time} seconds ({data_loading_time} seconds loading, {compute_time} seconds computing)")
