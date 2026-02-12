from hydra_zen import make_config, make_custom_builds_fn, zen, ZenStore
from hydra_zen.typing import Partial
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import numpy as np

from src.commons.mlp import MLP
from src.ls_mlp.data_driven_model import DataDrivenModel
from src.ls_mlp.data_module import DataModule
from src.ls_mlp.drift_model import DriftModel
from src.ls_mlp.plot_callback import PlotCallback
from src.ls_mlp.trainer_module import TrainerModule


"""
# Problem overview

## 1. Extension of the formulation of Rio *et al.* (2014)

We model the trajectory of a surface drifter as a stochastic differential equation (SDE):
$$
d\vec{X}_t = \left[ \vec{u}_g(t, \vec{X}_t) + \beta_e(t, \vec{X}_t) e^{i \theta_e(t, \vec{X}_t)} \vec{\tau}(t, \vec{X}_t) + \beta_w(t, \vec{X}_t) \vec{u}_w(t, \vec{X}_t) + \vec{\epsilon}(t, \vec{X}_t) \right] dt + \sigma d\vec{W}(t),
$$
where $\vec{X}_t\in\mathbb{R}^2$ denotes the horizontal position of the drifter at time $t$ and
$\vec{W}(t)$ is a two-dimensional standard Wiener process.

The terms in the drift component are defined as follows:
- $\vec{u}_g$ is the gridded geostrophic current velocity field,
- $\vec{\tau}$ is the gridded sea surface wind stress field,
- $\vec{u}_w$ is the gridded wind velocity field at 10 m above the sea surface,
- $\vec{\epsilon}$ is a space–time residual capturing unresolved processes,
- $\beta_e$ and $\theta_e$ are space–time continuous functions representing respectively the amplitude and the deflection angle of the empirical Ekman model from Rio *et al.* (2014),
- $\beta_w$ is a space–time continuous coefficient controlling the magnitude of the additional leeway contribution.

The scalar $\sigma$ controls the amplitude of the stochastic forcing, and accounts for unresolved subgrid-scale processes and model errors, we consider here $\sigma=1$.

## 2. Space–time continuous parametrization

The coefficients $\beta_e(t,\vec{X}_t)$, $\theta_e(t,\vec{X}_t)$, and $\beta_w(t,\vec{X}_t)$, and the residual $\vec{\epsilon}(t, \vec{X}_t)$ are modeled as smooth space–time continuous functions.
These functions are learned from observations using a multilayer perceptron (MLP) neural network.

The neural network takes transformed space–time coordinates derived from $(t, \vec{X}_t)$ as inputs and outputs the corresponding parameter values.
The network is trained using approximately 30 years of surface drifter observations, representing about $7\times 10^{7}$ position measurements at the global scale.

The time variable $t$ is expressed in encoded a the month-of-year. 
Periodic variables (month-of-year and longitude) are encoded using sine and cosine embeddings to capture their cyclical nature, this transformations enforces periodicity and avoids artificial discontinuities at coordinate boundaries.
Latitude is also encoded using sine and cosine transformations, to scale it in the interval $[-1,1]$ and to better represent distances between points on the sphere.
"""

EXP_ID = "least_squares_soft_physical_constraints"


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
        mlp=pbuilds(MLP, in_size=6, out_size=5, hidden_layers_size=[256, 512, 256, 128]),
        trainer_module=pbuilds(TrainerModule, optim_str="adamw", learning_rate=0.005),
        early_stop_callback = pbuilds(EarlyStopping, monitor="val", patience=5, min_delta=0., mode="min"),
        trainer = pbuilds(L.Trainer, accelerator="cpu", max_epochs=50, log_every_n_steps=10, enable_checkpointing=False)
    )

    return experiment_config


def main(
    data_module: Partial[DataModule],
    mlp: Partial[MLP],
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

    data_driven_model = DataDrivenModel(mlp=mlp())
    drift_model = DriftModel(
        data_driven_model=data_driven_model, 
        stress_normalization=jnp.ones(1), 
        wind_normalization=jnp.ones(1)
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
