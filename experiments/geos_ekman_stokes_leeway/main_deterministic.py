import os
os.environ["EQX_ON_ERROR"] = "nan"

import jax
jax.config.update("jax_enable_x64", True)

import dask
from hydra_zen import make_config, make_custom_builds_fn, zen, ZenStore
from hydra_zen.typing import Partial
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch

from src.data.datamodule import DataModule
from src.dynamics.linear_deterministic import LinearDeterministic
from src.module.deterministic import DeterministicModule


pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

ExperimentConfig = make_config(
    dynamics=pbuilds(
        LinearDeterministic.from_physical_space,
        depth_integrated_stokes=True, effective_wavenumber=True, include_leeway=True
    ),
    module=pbuilds(
        DeterministicModule,
        integration_dt=30*60,
        optimizer="rmsprop",
        learning_rate_scheduler="cosine",
        learning_rate=2e-3,
        loss_fn="separation_distance"
    ),
    datamodule=pbuilds(
        DataModule,
        train_test_val_splits=[0.64, 0.2, 0.16], batch_size=128, num_workers=32, prefetch_factor=3
    ),
    trainer=pbuilds(
        L.Trainer, 
        accelerator="auto", 
        logger=True,
        max_epochs=15, 
        enable_progress_bar=True, 
        log_every_n_steps=10
    ),
)


def do_calibrate(
    dynamics: Partial[LinearDeterministic],
    module: Partial[DeterministicModule],
    datamodule: Partial[DataModule], 
    trainer: Partial[L.Trainer],
    default_root_dir: str = None,
    dask_num_workers: int = 4
):
    torch.manual_seed(0)
    dask.config.set(scheduler="threads", num_workers=dask_num_workers)

    csv_logger = CSVLogger("lightning_logs", name="csv_logs")
    tb_logger = TensorBoardLogger("lightning_logs", name="tb_logs")

    module = module(dynamics=dynamics(), default_root_dir=default_root_dir)
    trainer = trainer(default_root_dir=default_root_dir, logger=[csv_logger, tb_logger])
    trainer.fit(module, datamodule=datamodule())


if __name__ == "__main__":
    store = ZenStore(deferred_hydra_store=False)
    store(ExperimentConfig, name="geos_ekman_stokes_leeway")

    zen(do_calibrate).hydra_main(
        config_name="geos_ekman_stokes_leeway",
        version_base="1.1",
        config_path=".",
    )
