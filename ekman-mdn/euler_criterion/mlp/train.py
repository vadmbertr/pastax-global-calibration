from hydra_zen import make_config, make_custom_builds_fn, zen, ZenStore
from hydra_zen.typing import Partial
import jax
jax.config.update("jax_enable_x64", True)
import jax.random as jrd
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import numpy as np

from src.commons.mlp import MLP
from src.ec_mlp.data_driven_model import DataDrivenModel
from src.ec_mlp.data_module import DataModule
from src.ec_mlp.drift_model import DriftModel
from src.ec_mlp.plot_callback import PlotCallback
from src.ec_mlp.trainer_module import TrainerModule


EXP_ID = "euler_criterion_soft_physical_constraints"


def create_default_config():
    pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

    from_datetime_str = "1994-06-01"
    to_datetime_str = "2025-08-01"

    experiment_config = make_config(
        data_module=pbuilds(
            DataModule,
            from_datetime_str=from_datetime_str,
            to_datetime_str=to_datetime_str,
            num_workers=16,
            multiprocessing_context="spawn",
            prefetch_factor=2,
            persistent_workers=True,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=0
        ),
        trunk=pbuilds(
            MLP, in_size=6, out_size=512, hidden_layers_size=[256, 512, 512], final_activation=jax.nn.gelu
        ),
        physical_head=pbuilds(MLP, in_size=512, out_size=3, hidden_layers_size=[128, 64]),
        mdn_head=pbuilds(MLP, in_size=512, out_size=32 * 6, hidden_layers_size=[512, 512, 512]),
        trainer_module=pbuilds(TrainerModule, optim_str="adamw", learning_rate=0.005),
        early_stop_callback = pbuilds(EarlyStopping, monitor="val", patience=5, min_delta=0., mode="min"),
        trainer = pbuilds(L.Trainer, accelerator="cpu", max_epochs=50, log_every_n_steps=10, enable_checkpointing=False)
    )

    return experiment_config


def main(
    data_module: Partial[DataModule],
    trunk: Partial[MLP],
    physical_head: Partial[MLP],
    mdn_head: Partial[MLP],
    trainer_module: Partial[TrainerModule],
    early_stop_callback: Partial[EarlyStopping],
    trainer: Partial[L.Trainer],
    data_path: str = "data/",
    batch_size: int = 2**14
):
    # the order of variables is important and must match the indexes used when manipulating batches
    var_names = [
        "month_of_year", "lat", "lon",
        "ve", "vn",
        "ugos", "vgos",
        "eastward_stress", "northward_stress",
        "eastward_wind", "northward_wind",
    ]
    # transformed variables are appended after regular variables, in the order defined in the transforms dict
    transforms = {
        "month_of_year": {"cos": lambda x: np.cos(2 * np.pi * x / 12), "sin": lambda x: np.sin(2 * np.pi * x / 12)},
        "lat": {"cos": lambda x: np.cos(np.deg2rad(x)), "sin": lambda x: np.sin(np.deg2rad(x))},
        "lon": {"cos": lambda x: np.cos(np.deg2rad(x)), "sin": lambda x: np.sin(np.deg2rad(x))}
    }
    # normalized variables are append after transformed variables, in the order defined in the normalize stats dict
    normalize_stats = {}

    data_module = data_module(
        data_path=data_path,
        var_names=var_names,
        transforms=transforms,
        normalize_stats=normalize_stats,
        batch_size=batch_size,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=0
    )

    key = jrd.key(0)
    trunk_key, physical_head_key, mdn_head_key = jrd.split(key, 3)
    trunk = trunk(key=trunk_key)
    physical_head = physical_head(key=physical_head_key)
    mdn_head = mdn_head(key=mdn_head_key)

    data_driven_model = DataDrivenModel(trunk, physical_head, mdn_head)
    drift_model = DriftModel(
        data_driven_model=data_driven_model, 
        stress_normalization=1., 
        wind_normalization=1.,
        delta_t=1.0 * 60.0 * 60.0  # 1 hour in seconds
    )

    trainer_module = trainer_module(drift_model=drift_model, batch_size=batch_size, exp_id=EXP_ID)

    csv_logger = CSVLogger("lightning_logs", name="csv_logs")
    tb_logger = TensorBoardLogger("lightning_logs", name="tb_logs")

    early_stop_callback = early_stop_callback()
    plot_callback = PlotCallback(data_path)

    trainer = trainer(callbacks=[early_stop_callback, plot_callback], logger=[csv_logger, tb_logger])

    trainer.fit(trainer_module, datamodule=data_module)


if __name__ == "__main__":
    store = ZenStore(deferred_hydra_store=False)
    store(create_default_config(), name=EXP_ID)

    zen(main).hydra_main(config_name=EXP_ID, version_base="1.1.0", config_path=".")
