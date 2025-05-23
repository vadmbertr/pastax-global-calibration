import os

import h5py
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .forcings import Duacs, Wave, Wind
from .gdp import GDP1h
from .xarray_dataset import XarrayDataset


def collate_fn(x):
    return x


class HDF5Data:
    id: str = "gdp-uc-uw-uh_idealized"

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

    def _create(self, gdp: GDP1h, uc: Duacs, uw: Wind, uh: Wave):
        def get_field_dim(f_sample):
            return len(f_sample[-3]), len(f_sample[-2]), len(f_sample[-1])
        
        def create_field_dataset(f, name, time_len, lat_len, lon_len):
            return f.create_dataset(
                name,
                (n_samples,),
                chunks=(1,),
                dtype=np.dtype(
                    [
                        ("u", "f4", ( time_len, lat_len, lon_len)), 
                        ("v", "f4", (time_len, lat_len, lon_len)), 
                        ("time", "i4", (time_len)), 
                        ("lat", "f4", (lat_len,)), 
                        ("lon", "f4", (lon_len,)), 
                    ]
                ),
                compression="lzf"
            )

        otf_ds = XarrayDataset(gdp, uc, uw, uh)

        n_samples = len(otf_ds)

        traj_sample, (uc_sample, uw_sample, uh_sample) = otf_ds[0]
        traj_len = len(traj_sample[0])
        uc_time_len, uc_lat_len, uc_lon_len = get_field_dim(uc_sample)
        uw_time_len, uw_lat_len, uw_lon_len = get_field_dim(uw_sample)
        uh_time_len, uh_lat_len, uh_lon_len = get_field_dim(uh_sample)
        
        otf_ds = XarrayDataset(gdp, uc, uw, uh)

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

            uc_ds = create_field_dataset(f, "uc", uc_time_len, uc_lat_len, uc_lon_len)
            uw_ds = create_field_dataset(f, "uw", uw_time_len, uw_lat_len, uw_lon_len)
            uh_ds = create_field_dataset(f, "uh", uh_time_len, uh_lat_len, uh_lon_len)

            for i, (gdp_sample, (uc_sample, uw_sample, uh_sample)) in enumerate(tqdm(otf_dl)):
                gdp_ds[i] = gdp_sample
                uc_ds[i] = uc_sample
                uw_ds[i] = uw_sample
                uh_ds[i] = uh_sample

    def prepare_data(self, gdp: GDP1h, uc: Duacs, uw: Wind, uh: Wave):
        gdp.prepare_data()
        uc.prepare_data()
        uw.prepare_data()
        uh.prepare_data()

        if not os.path.exists(f"{self.data_root}/{self.filename}"):
            self._create(gdp, uc, uw, uh)

    def __call__(self) -> h5py.File:
        return h5py.File(f"{self.data_root}/{self.filename}", "r")
