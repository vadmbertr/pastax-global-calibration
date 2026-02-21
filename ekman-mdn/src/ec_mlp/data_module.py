import os
from typing import Callable

import lightning.pytorch as L
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader

from src.commons.dataset import IncrementDataset, split_dataset


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        from_datetime_str: str,
        to_datetime_str: str,
        var_names: list[str],
        transforms: dict[str, dict[str, Callable]],
        normalize_stats: dict[str, tuple[float, float]],
        batch_size: int,
        num_workers: int = 8,
        multiprocessing_context: str = "spawn",
        prefetch_factor: int | None = None,
        persistent_workers: bool = True,
        val_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ):
        super().__init__()

        self._is_setup = False

        self.data_path = data_path
        self.from_datetime_str = from_datetime_str
        self.to_datetime_str = to_datetime_str

        self.var_names = var_names
        self.transforms = transforms
        self.normalize_stats = normalize_stats

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.multiprocessing_context = multiprocessing_context
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed

        self.u_x_normalization = None
        self.u_y_normalization = None
        self.stress_normalization = None
        self.wind_normalization = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if not os.path.exists(
            f"{self.data_path}/gdp_interp_clean_{self.from_datetime_str}_{self.to_datetime_str}_train.zarr"
        ):
            ds = xr.open_zarr(f"{self.data_path}/gdp_interp_clean_{self.from_datetime_str}_{self.to_datetime_str}.zarr")
            ds["month_of_year"] = ds.time.dt.month.astype(np.uint8)

            train_ds, val_ds, test_ds = split_dataset(
                ds, val_fraction=self.val_fraction, test_fraction=self.test_fraction, random_seed=self.seed,
            )

            chunks = ds.chunks["points"][0]

            train_ds.chunk({"points": chunks}).to_zarr(
                f"{self.data_path}/gdp_interp_clean_{self.from_datetime_str}_{self.to_datetime_str}_train.zarr"
            )
            val_ds.chunk({"points": chunks}).to_zarr(
                f"{self.data_path}/gdp_interp_clean_{self.from_datetime_str}_{self.to_datetime_str}_val.zarr"
            )
            test_ds.chunk({"points": chunks}).to_zarr(
                f"{self.data_path}/gdp_interp_clean_{self.from_datetime_str}_{self.to_datetime_str}_test.zarr"
            )

    def setup(self, stage=None):
        if self._is_setup:
            return
        
        train_ds = xr.open_zarr(
            f"{self.data_path}/gdp_interp_clean_{self.from_datetime_str}_{self.to_datetime_str}_train.zarr"
        )
        val_ds = xr.open_zarr(
            f"{self.data_path}/gdp_interp_clean_{self.from_datetime_str}_{self.to_datetime_str}_val.zarr"
        )
        test_ds = xr.open_zarr(
            f"{self.data_path}/gdp_interp_clean_{self.from_datetime_str}_{self.to_datetime_str}_test.zarr"
        )

        self.u_x_normalization = train_ds.ve.var().values.item() ** 0.5
        self.u_y_normalization = train_ds.vn.var().values.item() ** 0.5

        self.stress_normalization = np.sqrt(
            train_ds.eastward_stress.var() + train_ds.northward_stress.var()
        ).values.item()
        self.wind_normalization = np.sqrt(train_ds.eastward_wind.var() + train_ds.northward_wind.var()).values.item()
            
        def to_dataset(ds):
            return IncrementDataset(
                ds, var_names=self.var_names, transforms=self.transforms, normalize_stats=self.normalize_stats,
            )

        self.train_dataset = to_dataset(train_ds)
        self.val_dataset = to_dataset(val_ds)
        self.test_dataset = to_dataset(test_ds)

        self._is_setup = True

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False, persistent_workers=False)
    
    def predict_dataloader(self):
        return self.test_dataloader()

    def _loader(self, dataset, shuffle, persistent_workers):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            multiprocessing_context=self.multiprocessing_context,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=persistent_workers,
        )
