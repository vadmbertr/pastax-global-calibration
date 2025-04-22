import os

import h5py
import numpy as np
from tqdm import tqdm

from .forcings import Duacs
from .gdp import GDP1h
from .xarray_dataset import XarrayDataset


class HDF5Data:
    id: str = "gdp-uc"

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

    def _create(self, gdp: GDP1h, uc: Duacs):
        otf_ds = XarrayDataset(gdp, uc)

        n_samples = len(otf_ds)

        sample0 = otf_ds[0]
        traj_len = len(sample0[0][0])
        uc_time_len = len(sample0[1][-3])
        uc_lat_len = len(sample0[1][-2])
        uc_lon_len = len(sample0[1][-1])

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

            uc_ds = f.create_dataset(
                "uc", 
                (n_samples,),
                chunks=(1,),
                dtype=np.dtype(
                    [
                        ("u", "f4", ( uc_time_len, uc_lat_len, uc_lon_len)), 
                        ("v", "f4", (uc_time_len, uc_lat_len, uc_lon_len)), 
                        ("time", "i4", (uc_time_len)), 
                        ("lat", "f4", (uc_lat_len,)), 
                        ("lon", "f4", (uc_lon_len,)), 
                    ]
                ),
                compression="lzf"
            )

            for i, (gdp_sample, uc_sample) in enumerate(tqdm(otf_ds)):
                gdp_ds[i] = gdp_sample
                uc_ds[i] = uc_sample

    def prepare_data(self, gdp: GDP1h, uc: Duacs):
        gdp.prepare_data()
        uc.prepare_data()

        if not os.path.exists(f"{self.data_root}/{self.filename}"):
            self._create(gdp, uc)
        
    def __call__(self) -> h5py.File:
        return h5py.File(f"{self.data_root}/{self.filename}", "r")
