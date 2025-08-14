import os

import h5py
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .forcings import Duacs, Mur
from .gdp import GDP1h
from .xarray_dataset import XarrayDataset


def collate_fn(x):
    return x


class HDF5Data:
    id: str = "gdp-duacs-mur"

    def __init__(self, data_root: str, start_datetime: str | None, end_datetime: str | None):
        self.data_root = data_root
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

    @property
    def filename(self) -> str:
        filename = self.id
        if self.start_datetime is not None:
            filename += f"_{self.start_datetime}"
        if self.end_datetime is not None:
            filename += f"_{self.end_datetime}"
        return f"{filename}.hdf5"

    def _create(self, gdp: GDP1h, duacs: Duacs, mur: Mur):
        def get_field_dim(f_sample):
            return len(f_sample[-3]), len(f_sample[-2]), len(f_sample[-1])
        
        def create_field_dataset(f, name, field, time_len, lat_len, lon_len):
            dtype = [(k, "f4", (time_len, lat_len, lon_len)) for k in field.variables]
            dtype += [("time", "i4", (time_len,)), ("lat", "f4", (lat_len,)), ("lon", "f4", (lon_len,))]

            return f.create_dataset(
                name,
                (n_samples,),
                chunks=(1,),
                dtype=dtype,
                compression="lzf"
            )

        otf_ds = XarrayDataset(gdp, duacs, mur)

        n_samples = len(otf_ds)

        traj_sample, (duacs_sample, mur_sample) = otf_ds[0]
        traj_len = len(traj_sample[0])
        duacs_time_len, duacs_lat_len, duacs_lon_len = get_field_dim(duacs_sample)
        mur_time_len, mur_lat_len, mur_lon_len = get_field_dim(mur_sample)

        otf_ds = XarrayDataset(gdp, duacs, mur)

        otf_dl = DataLoader(
            otf_ds,
            batch_size=None,
            num_workers=10,
            pin_memory=False,
            prefetch_factor=1,
            multiprocessing_context="forkserver",
            collate_fn=collate_fn
        )

        with h5py.File(f"{self.data_root}/{self.filename}", "w") as f:
            gdp_ds = f.create_dataset(
                "gdp",
                (n_samples,),
                dtype=np.dtype(
                    [
                        ("lat", "f4", (traj_len,)), 
                        ("lon", "f4", (traj_len,)), 
                        ("time", "i4", (traj_len,)), 
                        ("id", "i4")
                    ]
                ),
                compression="lzf"
            )

            duacs_ds = create_field_dataset(f, "duacs", otf_ds.duacs, duacs_time_len, duacs_lat_len, duacs_lon_len)
            mur_ds = create_field_dataset(f, "mur", otf_ds.mur, mur_time_len, mur_lat_len, mur_lon_len)

            for i, (gdp_sample, (duacs_sample, mur_sample)) in enumerate(tqdm(otf_dl)):
                gdp_ds[i] = gdp_sample
                duacs_ds[i] = tuple([duacs_sample[0][k] for k in otf_ds.duacs.variables] + list(duacs_sample[1:]))
                mur_ds[i] = tuple([mur_sample[0][k] for k in otf_ds.mur.variables] + list(mur_sample[1:]))

    def prepare_data(self, gdp: GDP1h, duacs: Duacs, mur: Mur):
        if not os.path.exists(f"{self.data_root}/{self.filename}"):
            gdp.prepare_data()
            duacs.prepare_data()
            mur.prepare_data()
            
            self._create(gdp, duacs, mur)
        
    def __call__(self) -> h5py.File:
        return h5py.File(f"{self.data_root}/{self.filename}", "r")
