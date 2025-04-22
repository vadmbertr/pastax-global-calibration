import dask
from hydra_zen import builds, make_config, make_custom_builds_fn, zen, ZenStore
from hydra_zen.typing import Partial
import lightning as L

from src.data.datamodule import DataModule
from src.dynamics.linear_deterministic import LinearDeterministic
from src.module.deterministic import DeterministicModule


pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

ExperimentConfig = make_config(
    datamodule=builds(
        DataModule, 
        start_datetime=None, end_datetime=None,
        train_test_val_splits=[0.64, 0.2, 0.16], batch_size=128, num_workers=24, prefetch_factor=2
    ),
    dynamics=builds(LinearDeterministic, intercept=0, slope=1),
    module=pbuilds(DeterministicModule, integration_horizon=5, integration_dt=30*60, learning_rate=1e-3),
    checkpointer=builds(L.pytorch.callbacks.ModelCheckpoint, monitor="val_loss", mode="min", save_top_k=1),
    trainer=pbuilds(L.Trainer, accelerator="auto", logger=True, max_epochs=10, enable_progress_bar=True),
)


def do_calibrate(
    datamodule: DataModule, 
    dynamics: LinearDeterministic,
    module: Partial[DeterministicModule],
    checkpointer: L.pytorch.callbacks.ModelCheckpoint,
    trainer: Partial[L.Trainer],
    jax_enable_x64: bool = True,
    dask_num_workers: int = 4
):  
    if jax_enable_x64:
        import jax
        jax.config.update("jax_enable_x64", True)
    
    dask.config.set(scheduler="threads", num_workers=dask_num_workers)

    module = module(dynamics=dynamics)

    trainer = trainer(callbacks=[checkpointer])
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    store = ZenStore(deferred_hydra_store=False)
    store(ExperimentConfig, name="geos")

    zen(do_calibrate).hydra_main(
        config_name="geos",
        version_base="1.1",
        config_path=".",
    )
