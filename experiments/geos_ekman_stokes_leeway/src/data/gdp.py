import clouddrift as cd
import numpy as np
import xarray as xr

from .zarr_data import ZarrData


class GDP1h(ZarrData):
    id: str = "noaa-oar-hourly-gdp-pds"

    def _download(self):
        ds = cd.datasets.gdp1h()
        
        ds.time.load()
        ds = cd.ragged.subset(
            ds, 
            {"time": lambda t: (t >= np.datetime64(self.start_datetime)) and (t < np.datetime64(self.end_datetime))}, 
            row_dim_name="traj"
        )

        ds = ds[["drogue_status", "time", "lat", "lon", "vn", "ve", "err_lat", "err_lon", "rowsize"]]
        
        ds = restrict_to_drogued(ds)
        ds = remove_outliers(ds)
        ds = create_chuncks(ds)
        
        ds.to_zarr(f"{self.data_root}/{self.filename}")


def restrict_to_drogued(ds: xr.Dataset) -> xr.Dataset:
    ds.drogue_status.load()
    return cd.ragged.subset(ds, {"drogue_status": True}, row_dim_name="traj")


def remove_outliers(ds: xr.Dataset, velocity_cutoff: float = 10, latlon_err_cutoff: float = .5) -> xr.Dataset:
    # remove nans
    ds.lat.load()
    ds = cd.ragged.subset(ds, {"lat": np.isfinite}, row_dim_name="traj")
    ds.lon.load()
    ds = cd.ragged.subset(ds, {"lon": np.isfinite}, row_dim_name="traj")
    ds.vn.load()
    ds = cd.ragged.subset(ds, {"vn": np.isfinite}, row_dim_name="traj")
    ds.ve.load()
    ds = cd.ragged.subset(ds, {"ve": np.isfinite}, row_dim_name="traj")
    ds.time.load()
    ds = cd.ragged.subset(ds, {"time": lambda arr: ~np.isnat(arr)}, row_dim_name="traj")

    # remove outliers
    ds.vn.load()
    ds = cd.ragged.subset(ds, {"vn": (-velocity_cutoff, velocity_cutoff)}, row_dim_name="traj")
    ds.ve.load()
    ds = cd.ragged.subset(ds, {"ve": (-velocity_cutoff, velocity_cutoff)}, row_dim_name="traj")
    ds.err_lat.load()
    ds = cd.ragged.subset(ds, {"err_lat": (0, latlon_err_cutoff)}, row_dim_name="traj")
    ds.err_lon.load()
    ds = cd.ragged.subset(ds, {"err_lon": (0, latlon_err_cutoff)}, row_dim_name="traj")

    return ds


def create_chuncks(ds: xr.Dataset, n_days: int = 5, dt: np.timedelta64 = None, to_ragged: bool = False) -> xr.Dataset:
    def ragged_chunk(arr: xr.DataArray | np.ndarray, is_metadata: bool = False) -> np.ndarray:
        arr = cd.ragged.apply_ragged(cd.ragged.chunk, arr, row_size, chunk_size)  # noqa
        if is_metadata:
            arr = arr[:, 0]
        if to_ragged:
            arr = arr.ravel()
        return arr

    if dt is None:
        dt = (ds.isel(traj=0).time[1] - ds.isel(traj=0).time[0]).values

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
