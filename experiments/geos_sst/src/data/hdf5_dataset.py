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
        tuple[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ]
    ]:
        if self.data_file is None:
            self.data_file = self.data()
            
        traj_lat, traj_lon, traj_time, traj_id = self.data_file["gdp"][idx]
        duacs_u, duacs_v, duacs_time, duacs_lat, duacs_lon = self.data_file["duacs"][idx]
        mur_sst, mur_time, mur_lat, mur_lon = self.data_file["mur"][idx]

        return (
            traj_lat, traj_lon, traj_time, traj_id
        ), (
            (duacs_u, duacs_v, duacs_time, duacs_lat, duacs_lon),
            (mur_sst, mur_time, mur_lat, mur_lon),
        )
