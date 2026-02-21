from pathlib import Path
from typing import Callable

from jaxtyping import Float
import numpy as np
from torch.utils.data import Dataset, get_worker_info
import xarray as xr


def split_dataset(
    ds: xr.Dataset, val_fraction: float = 0.2, test_fraction: float = 0.2, random_seed: int = 0
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    rng = np.random.default_rng(random_seed)

    drifters_id = np.unique(ds.id)
    num_drifters = len(drifters_id)

    indices = np.arange(num_drifters)
    rng.shuffle(indices)

    val_size = int(num_drifters * val_fraction)
    test_size = int(num_drifters * test_fraction)
    train_size = num_drifters - val_size - test_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]

    is_train = np.isin(ds.id, drifters_id[train_indices])
    ds_train = ds.where(
        xr.DataArray(data=is_train).rename({"dim_0": "points"}), drop=True
    )
    is_val = (~is_train) & np.isin(ds.id, drifters_id[val_indices])
    ds_val = ds.where(
        xr.DataArray(data=is_val).rename({"dim_0": "points"}), drop=True
    )
    is_test = (~is_train) & (~is_val)
    ds_test = ds.where(
        xr.DataArray(data=is_test).rename({"dim_0": "points"}), drop=True
    )

    return ds_train, ds_val, ds_test


class IncrementDataset(Dataset):
    def __init__(
        self,
        ds: xr.Dataset,
        var_names: list[str], 
        transforms: dict[str, dict[str, Callable]] = {},
        normalize_stats: dict[str, tuple[float, float]] = {}
    ):
        ds = ds[var_names].load()
        features = np.stack([ds[var_name].values for var_name in var_names], axis=-1)
        transformed_features = self.__compute_transformed_features(ds, transforms)
        normalized_features = self.__compute_normalized_features(ds, normalize_stats)

        all_features = [features]
        if transformed_features is not None:
            all_features.append(transformed_features)
        if normalized_features is not None:
            all_features.append(normalized_features)
        
        self.data = np.concatenate(all_features, axis=-1)

        self.length = self.data.shape[0]

    @staticmethod
    def __compute_transformed_features(
        ds: xr.Dataset, transforms: dict[str, dict[str, Callable]]
    ) -> np.ndarray | None:
        features = []

        for name, transform_dict in transforms.items():
            value = ds[name].values

            for transform in transform_dict.values():
                features.append(np.asarray(transform(value), dtype=value.dtype))

        if features:
            features = np.stack(features, axis=-1)
        else:
            features = None

        return features

    @staticmethod
    def __compute_normalized_features(
        ds: xr.Dataset, normalize_stats: dict[str, tuple[float, float]]
    ) -> np.ndarray | None:
        features = []

        for name, (mean, std) in normalize_stats.items():
            value = ds[name].values
            norm = (value - mean) / (std + 1e-6)
            features.append(np.asarray(norm, dtype=value.dtype))

        if features:
            features = np.stack(features, axis=-1)
        else:
            features = None

        return features

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.open_zarr()


class TrajectoryDataset(Dataset):
    def __init__(self, traj_path: Path, duacs_path: Path, era5_path: Path, periodic_domain: bool = True):
        self.traj_path = traj_path
        self.duacs_path = duacs_path
        self.era5_path = era5_path
        self.periodic_domain = periodic_domain

        traj_ds = xr.open_zarr(self.traj_path)
        duacs_ds = xr.open_zarr(self.duacs_path)
        era5_ds = xr.open_zarr(self.era5_path)

        max_travel_distance = .5  # in ° / day ; inferred from data
        traj_t0_t1 = traj_ds.time.isel(traj=0)[np.asarray([0, -1])]
        n_days = ((traj_t0_t1[-1] - traj_t0_t1[0]) / np.timedelta64(1, "D")).astype(int).values.item()
        max_travel_distance *= n_days

        self.duacs_nt = duacs_ds.time.size
        self.duacs_mint = duacs_ds.time.min()
        self.duacs_dt = duacs_ds.time[1] - duacs_ds.time[0]
        self.duacs_t_di = np.ceil(np.timedelta64(n_days, "D") / self.duacs_dt)
        self.duacs_nlat = duacs_ds.latitude.size
        self.duacs_minlat = duacs_ds.latitude.min()
        self.duacs_dlat = duacs_ds.latitude[1] - duacs_ds.latitude[0]  # regular grid
        self.duacs_lat_di = np.ceil(max_travel_distance / self.duacs_dlat)
        self.duacs_nlon = duacs_ds.longitude.size
        self.duacs_minlon = duacs_ds.longitude.min()
        self.duacs_dlon = duacs_ds.longitude[1] - duacs_ds.longitude[0]  # regular grid
        self.duacs_lon_di = np.ceil(max_travel_distance / self.duacs_dlon)

        self.era5_nt = era5_ds.time.size
        self.era5_mint = era5_ds.time.min()
        self.era5_dt = era5_ds.time[1] - era5_ds.time[0]
        self.era5_t_di = np.ceil(np.timedelta64(n_days, "D") / self.era5_dt)
        self.era5_nlat = era5_ds.latitude.size
        self.era5_minlat = era5_ds.latitude.min()
        self.era5_dlat = era5_ds.latitude[1] - era5_ds.latitude[0]  # regular grid
        self.era5_lat_di = np.ceil(max_travel_distance / self.era5_dlat)
        self.era5_nlon = era5_ds.longitude.size
        self.era5_minlon = era5_ds.longitude.min()
        self.era5_dlon = era5_ds.longitude[1] - era5_ds.longitude[0]  # regular grid
        self.era5_lon_di = np.ceil(max_travel_distance / self.era5_dlon)

        self.len = traj_ds.traj.size

        self.traj_ds = None
        self.duacs_ds = None
        self.era5_ds = None
        self.opened = False

    def __len__(self) -> int:
        return self.len

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[
            Float[np.ndarray, "traj_length"], 
            Float[np.ndarray, "traj_length"], 
            Float[np.ndarray, "traj_length"]
        ],
        tuple[
            dict[str, Float[np.ndarray, "T N N"]], 
            Float[np.ndarray, "T"], 
            Float[np.ndarray, "N"], 
            Float[np.ndarray, "N"]
        ],
        tuple[
            dict[str, Float[np.ndarray, "T N N"]], 
            Float[np.ndarray, "T"], 
            Float[np.ndarray, "N"], 
            Float[np.ndarray, "N"]
        ]
    ]:
        self.open_zarr()

        traj_arrays = None
        duacs_arrays = None
        era5_arrays = None
        while True:
            try:
                if traj_arrays is None:
                    traj_arrays = self.__get_traj_arrays(idx)
                if duacs_arrays is None:
                    duacs_arrays = self.__get_duacs_arrays(*traj_arrays)
                if era5_arrays is None:
                    era5_arrays = self.__get_era5_arrays(*traj_arrays)
            except Exception as e:
                print(f"Error while loading sample {idx}: {e}. Retrying...")
            else:
                break
        
        return traj_arrays, duacs_arrays, era5_arrays

    def open_zarr(self):
        if not self.opened:
            self.traj_ds = xr.open_zarr(self.traj_path)
            self.duacs_ds = xr.open_zarr(self.duacs_path)
            self.era5_ds = xr.open_zarr(self.era5_path)
            self.opened = True
    
    def __get_traj_arrays(
        self, idx: int
    ) -> tuple[
        Float[np.ndarray, "traj_length"], 
        Float[np.ndarray, "traj_length"], 
        Float[np.ndarray, "traj_length"]
    ]:
        traj_subset = self.traj_ds.isel(traj=idx)
        
        traj_lat = traj_subset.lat.values.ravel()
        traj_lon = traj_subset.lon.values.ravel()
        traj_time = traj_subset.time.values.ravel().astype("datetime64[s]").astype(int)  # in seconds
        
        return traj_lat, traj_lon, traj_time
    
    def __get_duacs_arrays(
        self, 
        traj_lat: Float[np.ndarray, "traj_length"], 
        traj_lon: Float[np.ndarray, "traj_length"], 
        traj_time: Float[np.ndarray, "traj_length"]
    ) -> tuple[
        dict[str, Float[np.ndarray, "T N N"]], 
        Float[np.ndarray, "T"], 
        Float[np.ndarray, "N"], 
        Float[np.ndarray, "N"]
    ]:
        return self.__get_forcing_arrays(
            traj_lat, traj_lon, traj_time,
            self.duacs_ds, ("ugos", "vgos"),
            self.duacs_nt, self.duacs_mint, self.duacs_dt, self.duacs_t_di,
            self.duacs_nlat, self.duacs_minlat, self.duacs_dlat, self.duacs_lat_di,
            self.duacs_nlon, self.duacs_minlon, self.duacs_dlon, self.duacs_lon_di
        )
    
    def __get_era5_arrays(
        self, 
        traj_lat: Float[np.ndarray, "traj_length"], 
        traj_lon: Float[np.ndarray, "traj_length"], 
        traj_time: Float[np.ndarray, "traj_length"]
    ) -> tuple[
        dict[str, Float[np.ndarray, "T N N"]], 
        Float[np.ndarray, "T"], 
        Float[np.ndarray, "N"], 
        Float[np.ndarray, "N"]
    ]:
        return self.__get_forcing_arrays(
            traj_lat, traj_lon, traj_time,
            self.era5_ds, ("eastward_stress", "northward_stress", "eastward_wind", "northward_wind"),
            self.era5_nt, self.era5_mint, self.era5_dt, self.era5_t_di,
            self.era5_nlat, self.era5_minlat, self.era5_dlat, self.era5_lat_di,
            self.era5_nlon, self.era5_minlon, self.era5_dlon, self.era5_lon_di
        )
    
    def __get_forcing_arrays(
        self, 
        traj_lat: Float[np.ndarray, "traj_length"], 
        traj_lon: Float[np.ndarray, "traj_length"], 
        traj_time: Float[np.ndarray, "traj_length"],
        ds: xr.Dataset, 
        vars_names: tuple[str],
        nt: int, mint: np.datetime64, dt: np.timedelta64, t_di: int, 
        nlat: int, minlat: float, dlat: float, lat_di: float, 
        nlon: int, minlon: float, dlon: float, lon_di: float
    ) -> tuple[
        dict[str, Float[np.ndarray, "T N N"]], 
        Float[np.ndarray, "T"], 
        Float[np.ndarray, "N"], 
        Float[np.ndarray, "N"]
    ]:
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

        t0_i = np.floor((t0 - mint) / dt)
        lat0_i = ((lat0 - minlat) / dlat).round()
        lon0_i = ((lon0 - minlon) / dlon).round()

        tmin_i = t0_i.astype(int).item()
        tmax_i = (t0_i + t_di).astype(int).item()
        latmin_i, latmax_i = get_latlon_minmax(lat0_i, lat_di)
        lonmin_i, lonmax_i = get_latlon_minmax(lon0_i, lon_di)

        (t_padleft, t_padright), (tmin_i, tmax_i) = get_pads(tmin_i, tmax_i, nt)
        (lat_padleft, lat_padright), (latmin_i, latmax_i) = get_pads(latmin_i, latmax_i, nlat)
        (lon_padleft, lon_padright), (lonmin_i, lonmax_i) = get_pads(lonmin_i, lonmax_i, nlon)

        patch = ds.isel(
            time=slice(tmin_i, tmax_i + 1),
            latitude=slice(latmin_i, latmax_i + 1), 
            longitude=slice(lonmin_i, lonmax_i + 1)
        )

        patch_vars = dict((var_name, patch[var_name]) for var_name in vars_names)
        patch_time = patch.time.astype("datetime64[s]").astype(int)  # in seconds
        patch_lat = patch.latitude
        patch_lon = patch.longitude

        if self.periodic_domain:  # periodic global domain
            if lon_padleft != 0:
                patch_left = ds.isel(
                    time=slice(tmin_i, tmax_i + 1),
                    latitude=slice(latmin_i, latmax_i + 1), 
                    longitude=slice(nlon - lon_padleft, nlon)  # right part goes to the left
                )

                patch_vars_left = dict((var, patch_left[var]) for var in vars_names)
                lon_left = patch_left.longitude

                patch_vars = dict(
                    (var_name, np.concat([patch_vars_left[var_name], patch_vars[var_name]], axis=-1)) 
                    for var_name in vars_names
                )
                patch_lon = np.concat([lon_left, patch_lon])

                lon_padleft = 0

            if lon_padright != 0:
                patch_right = ds.isel(
                    time=slice(tmin_i, tmax_i + 1),
                    latitude=slice(latmin_i, latmax_i + 1), 
                    longitude=slice(0, lon_padright)  # left part goes to the right
                )

                patch_vars_right = dict((var, patch_right[var]) for var in vars_names)
                lon_right = patch_right.longitude

                patch_vars = dict(
                    (var_name, np.concat([patch_vars[var_name], patch_vars_right[var_name]], axis=-1)) 
                    for var_name in vars_names
                )
                patch_lon = np.concat([patch_lon, lon_right])

                lon_padright = 0

        patch_vars = dict(
            (
                var_name,
                np.pad(
                    patch_vars[var_name], 
                    ((t_padleft, t_padright), (lat_padleft, lat_padright), (lon_padleft, lon_padright)), 
                    mode="edge"
                )
            ) for var_name in vars_names
        )
        patch_time = np.pad(patch_time, (t_padleft, t_padright), mode="edge")
        patch_lat = np.pad(patch_lat, (lat_padleft, lat_padright), mode="edge")
        patch_lon = np.pad(patch_lon, (lon_padleft, lon_padright), mode="edge")
        
        return patch_vars, patch_time, patch_lat, patch_lon
