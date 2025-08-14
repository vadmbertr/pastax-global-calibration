import numpy as np
from torch.utils.data import Dataset as TorchDataset

from .forcings import Duacs, Mur
from .gdp import GDP1h
from .zarr_data import ZarrData


class _PhysicalField:
    def __init__(
        self, data: ZarrData, variables: dict[str, str], periodic_grid: bool, max_travel_distance: float, n_days: int
    ):
        self.variables = variables
        self.periodic_grid = periodic_grid

        self.ds = data()

        self.data = data
        self.ds = None
        
        self.nt = self.ds.time.size
        self.mint = self.ds.time.min()
        self.dt = self.ds.time[1] - self.ds.time[0]
        self.t_di = np.ceil(np.timedelta64(n_days, "D") / self.dt)
        self.nlat = self.ds.latitude.size
        self.minlat = self.ds.latitude.min()
        self.dlat = self.ds.latitude[1] - self.ds.latitude[0]  # regular grid
        self.lat_di = np.ceil(max_travel_distance / self.dlat)
        self.nlon = self.ds.longitude.size
        self.minlon = self.ds.longitude.min()
        self.dlon = self.ds.longitude[1] - self.ds.longitude[0]  # regular grid
        self.lon_di = np.ceil(max_travel_distance / self.dlon)

    def __call__(
        self, traj_lat: np.ndarray, traj_lon: np.ndarray, traj_time: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        t0_i = np.floor((t0 - self.mint) / self.dt)
        lat0_i = ((lat0 - self.minlat) / self.dlat).round()
        lon0_i = ((lon0 - self.minlon) / self.dlon).round()

        tmin_i = t0_i.astype(int).item()
        tmax_i = (t0_i + self.t_di).astype(int).item()
        latmin_i, latmax_i = get_latlon_minmax(lat0_i, self.lat_di)
        lonmin_i, lonmax_i = get_latlon_minmax(lon0_i, self.lon_di)

        (t_padleft, t_padright), (tmin_i, tmax_i) = get_pads(tmin_i, tmax_i, self.nt)
        (lat_padleft, lat_padright), (latmin_i, latmax_i) = get_pads(latmin_i, latmax_i, self.nlat)
        (lon_padleft, lon_padright), (lonmin_i, lonmax_i) = get_pads(lonmin_i, lonmax_i, self.nlon)
        
        patch = self.ds.isel(
            time=slice(tmin_i, tmax_i + 1),
            latitude=slice(latmin_i, latmax_i + 1), 
            longitude=slice(lonmin_i, lonmax_i + 1)
        )

        variables = {k: patch[self.variables[k]] for k in self.variables}
        time = patch.time.astype("datetime64[s]").astype(int)  # in seconds
        lat = patch.latitude
        lon = patch.longitude

        if self.periodic_grid:  # periodic global domain
            if lon_padleft != 0:
                patch_left = self.ds.isel(
                    time=slice(tmin_i, tmax_i + 1),
                    latitude=slice(latmin_i, latmax_i + 1), 
                    longitude=slice(self.nlon - lon_padleft, self.nlon)  # right part goes to the left
                )
                variables_left = {k: patch_left[self.variables[k]] for k in self.variables}
                lon_left = patch_left.longitude

                variables = {k: np.concat([variables_left[k], v], axis=-1) for k, v in variables.items()}
                lon = np.concat([lon_left, lon])

                lon_padleft = 0

            if lon_padright != 0:
                patch_right = self.ds.isel(
                    time=slice(tmin_i, tmax_i + 1),
                    latitude=slice(latmin_i, latmax_i + 1), 
                    longitude=slice(0, lon_padright)  # left part goes to the right
                )
                variables_right = {k: patch_right[self.variables[k]] for k in self.variables}
                lon_right = patch_right.longitude

                variables = {k: np.concat([v, variables_right[k]], axis=-1) for k, v in variables.items()}
                lon = np.concat([lon, lon_right])

                lon_padright = 0

        variables = {
            k: np.pad(
                v,
                ((t_padleft, t_padright), (lat_padleft, lat_padright), (lon_padleft, lon_padright)), 
                mode="edge"
            )
            for k, v in variables.items()
        }
        time = np.pad(time, (t_padleft, t_padright), mode="edge")
        lat = np.pad(lat, (lat_padleft, lat_padright), mode="edge")
        lon = np.pad(lon, (lon_padleft, lon_padright), mode="edge")

        return variables, time, lat, lon


class XarrayDataset(TorchDataset):
    def __init__(
        self,
        gdp: GDP1h, 
        duacs: Duacs,
        mur: Mur,
        periodic_grids: bool = True
    ):
        self.traj_ds = gdp()

        self.size = traj_ds.traj.size
        self.gdp = gdp
        self.traj_ds = None

        max_travel_distance = .5  # in ° / day ; inferred from data
        traj_t0_t1 = self.traj_ds.time.isel(traj=0)[np.asarray([0, -1])]
        n_days = ((traj_t0_t1[-1] - traj_t0_t1[0]) / np.timedelta64(1, "D")).astype(int).as_numpy().item()
        max_travel_distance = max_travel_distance * n_days

        self.duacs = _PhysicalField(duacs, {"u": "ugos", "v": "vgos"}, periodic_grids, max_travel_distance, n_days)
        self.mur = _PhysicalField(mur, {"T": "analysed_sst"}, periodic_grids, max_travel_distance, n_days)

    def __len__(self) -> int:
        return self.traj_ds.traj.size

    def __getitem__(self, idx: int) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        traj_arrays = self.__get_traj_arrays(idx)
        fields_arrays = self.__get_fields_arrays(*traj_arrays[:3])

        return traj_arrays, fields_arrays
    
    def __get_traj_arrays(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.traj_ds is None:
            self.traj_ds = self.gdp()
        
        traj_subset = self.traj_ds.isel(traj=idx)
        
        traj_lat = traj_subset.lat.values.ravel()
        traj_lon = traj_subset.lon.values.ravel()
        traj_time = traj_subset.time.values.ravel().astype("datetime64[s]").astype(int)  # in seconds
        traj_id = traj_subset.id.values.ravel()
        
        return traj_lat, traj_lon, traj_time, traj_id
    
    def __get_fields_arrays(
        self, traj_lat: np.ndarray, traj_lon: np.ndarray, traj_time: np.ndarray
    ) -> tuple[
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray],
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]
    ]:
        duacs = self.duacs(traj_lat, traj_lon, traj_time)
        mur = self.mur(traj_lat, traj_lon, traj_time)

        return duacs, mur
