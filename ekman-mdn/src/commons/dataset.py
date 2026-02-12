from typing import Callable

import numpy as np
from torch.utils.data import Dataset
import xarray as xr


def split_dataset(
    ds: xr.Dataset, val_fraction: float = 0.2, test_fraction: float = 0.2, random_seed: int = 0
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    rng = np.random.default_rng(random_seed)

    drifters_id = np.unique(ds.id)
    num_drifters = len(drifters_id)

    indices = np.arange(num_drifters)
    rng.shuffle(indices)

    val_size = int(num_drifters * val_fraction)
    test_size = int(num_drifters * test_fraction)
    train_size = num_drifters - val_size - test_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    is_train = np.isin(ds.id, drifters_id[train_indices])
    ds_train = ds.where(
        xr.DataArray(data=is_train).rename({"dim_0": "points"}), drop=True
    )
    is_val = (~is_train) & np.isin(ds.id, drifters_id[val_indices])
    ds_val = ds.where(
        xr.DataArray(data=is_val).rename({"dim_0": "points"}), drop=True
    )
    is_test = (~is_train) & (~is_val)
    ds_test = ds.where(
        xr.DataArray(data=is_test).rename({"dim_0": "points"}), drop=True
    )

    return ds_train, ds_val, ds_test


class XarrayDataset(Dataset):
    def __init__(
        self,
        ds: xr.Dataset,
        var_names: list[str], 
        transforms: dict[str, dict[str, Callable]] = {},
        normalize_stats: dict[str, tuple[float, float]] = {}
    ):
        ds = ds[var_names].load()
        features = np.stack([ds[var_name].values for var_name in var_names], axis=-1)
        transformed_features = self.__compute_transformed_features(ds, transforms)
        normalized_features = self.__compute_normalized_features(ds, normalize_stats)

        all_features = [features]
        if transformed_features is not None:
            all_features.append(transformed_features)
        if normalized_features is not None:
            all_features.append(normalized_features)
        
        self.data = np.concatenate(all_features, axis=-1)

        self.length = self.data.shape[0]

    @staticmethod
    def __compute_transformed_features(
        ds: xr.Dataset, transforms: dict[str, dict[str, Callable]]
    ) -> np.ndarray | None:
        features = []

        for name, transform_dict in transforms.items():
            value = ds[name].values

            for transform in transform_dict.values():
                features.append(np.asarray(transform(value), dtype=value.dtype))

        if features:
            features = np.stack(features, axis=-1)
        else:
            features = None

        return features

    @staticmethod
    def __compute_normalized_features(
        ds: xr.Dataset, normalize_stats: dict[str, tuple[float, float]]
    ) -> np.ndarray | None:
        features = []

        for name, (mean, std) in normalize_stats.items():
            value = ds[name].values
            norm = (value - mean) / (std + 1e-6)
            features.append(np.asarray(norm, dtype=value.dtype))

        if features:
            features = np.stack(features, axis=-1)
        else:
            features = None

        return features

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]
