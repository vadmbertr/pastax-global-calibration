import dask
from hydra_zen import builds, make_config, make_custom_builds_fn, zen, ZenStore
from hydra_zen.typing import Partial
import lightning as L
import torch

from src.data.datamodule import DataModule
from src.dynamics.linear_stochastic import LinearStochastic
from src.module.stochastic import StochasticModule


pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

ExperimentConfig = make_config(
    datamodule=builds(
        DataModule,
        train_test_val_splits=[0.64, 0.2, 0.16], batch_size=128, num_workers=32, prefetch_factor=3
    ),
    module=pbuilds(
        StochasticModule, 
        integration_dt=30*60, 
        optimizer="rmsprop", 
        learning_rate_scheduler="cosine", 
        learning_rate=1e-3, 
        antithetic_variate=True,
        loss_fn="separation_distance"
    ),
    checkpointer=builds(L.pytorch.callbacks.ModelCheckpoint, monitor="val_loss", mode="min", save_top_k=1),
    trainer=pbuilds(
        L.Trainer, 
        accelerator="auto", 
        logger=True,
        max_epochs=10, 
        enable_progress_bar=True, 
        log_every_n_steps=1
    ),
)


def do_calibrate(
    datamodule: DataModule, 
    module: Partial[StochasticModule],
    checkpointer: L.pytorch.callbacks.ModelCheckpoint,
    trainer: Partial[L.Trainer],
    jax_enable_x64: bool = True,
    dask_num_workers: int = 4
):
    if jax_enable_x64:
        import jax
        jax.config.update("jax_enable_x64", True)

    torch.manual_seed(0)
    dask.config.set(scheduler="threads", num_workers=dask_num_workers)

    dynamics = LinearStochastic.from_physical_space()
    module = module(dynamics=dynamics)
    trainer = trainer(callbacks=[checkpointer])
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    store = ZenStore(deferred_hydra_store=False)
    store(ExperimentConfig, name="geos_ekman_stokes_leeway")

    zen(do_calibrate).hydra_main(
        config_name="geos_ekman_stokes_leeway",
        version_base="1.1",
        config_path=".",
    )
