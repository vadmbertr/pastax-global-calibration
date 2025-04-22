import numpy as np
from torch.utils.data import Dataset as TorchDataset

from .hdf5_data import HDF5Data


class HDF5Dataset(TorchDataset):
    def __init__(self, data: HDF5Data
    ):
        self.data = data
        with self.data() as f:
            self.size = f["gdp"].shape[0]
        self.data_file = None

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        if self.data_file is None:
            self.data_file = self.data()
            
        traj_lat, traj_lon, traj_time, traj_id = self.data_file["gdp"][idx]
        uc_u, uc_v, uc_time, uc_lat, uc_lon = self.data_file["uc"][idx]

        return (traj_lat, traj_lon, traj_time, traj_id), (uc_u, uc_v, uc_time, uc_lat, uc_lon)
