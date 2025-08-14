from __future__ import annotations
import os

import xarray as xr


class ZarrData:
    id: str = None

    def __init__(self, data_root: str, start_datetime: str, end_datetime: str):
        self.data_root = data_root
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

    @property
    def filename(self) -> str:
        return f"{self.id}_{self.start_datetime}_{self.end_datetime}.zarr"

    def _download(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def prepare_data(self):
        if not os.path.exists(f"{self.data_root}/{self.filename}"):
            self._download()

    def __call__(self) -> xr.Dataset:
        ds = xr.open_zarr(f"{self.data_root}/{self.filename}")
        return ds
