import lightning as L
import torch
from torch.utils.data import DataLoader, random_split, Subset

from .forcings import Duacs, Wave, Wind
from .gdp import GDP1h
from .hdf5_data import HDF5Data
from .hdf5_dataset import HDF5Dataset


class DataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_root: str, 
        start_datetime: str | None, 
        end_datetime: str | None, 
        train_test_val_splits: list[float], 
        batch_size: int,
        num_workers: int,
        prefetch_factor: int
    ):
        super().__init__()

        self.gdp = GDP1h(data_root, start_datetime, end_datetime)
        self.uc = Duacs(data_root, start_datetime, end_datetime)
        self.uw = Wind(data_root, start_datetime, end_datetime)
        self.uh = Wave(data_root, start_datetime, end_datetime)

        self.hdf5_data = HDF5Data(data_root, start_datetime, end_datetime)

        self.train_test_val_splits = train_test_val_splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        self.hdf5_data.prepare_data(self.gdp, self.uc, self.uw, self.uh)

    def setup(self, stage: str = None):
        dataset = HDF5Dataset(self.hdf5_data)

        train_dataset, test_dataset, val_dataset = random_split(
            dataset, self.train_test_val_splits, generator=torch.Generator().manual_seed(0)
        )

        if stage == "fit":
            self.train_dataset, self.val_dataset = train_dataset, val_dataset
            # self.train_dataset, self.val_dataset = Subset(train_dataset, [0]), Subset(val_dataset, [0])
        if stage == "test":
            self.test_dataset = test_dataset
        if stage == "predict":
            self.predict_dataset = test_dataset

    def train_dataloader(self):
        return self.__return_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.__return_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.__return_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self.__return_dataloader(self.predict_dataset, shuffle=False)
    
    def __return_dataloader(self, dataset: HDF5Dataset, shuffle: bool):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            in_order=False
        )
