import io
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
import lightning.pytorch as L
import numpy as np
import optax
import torch

from src.ec_mlp.drift_model import DriftModel


class TrainerModule(L.LightningModule):
    def __init__(
        self, 
        drift_model: DriftModel,
        optim_str: Literal["adam", "adamw"],
        learning_rate: float,
        batch_size: int,
        exp_id: str
    ):
        super().__init__()
        
        self.automatic_optimization = False

        self.drift_model = drift_model
        self.u_x_normalization = None
        self.u_y_normalization = None

        self.optim_str = optim_str
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.exp_id = exp_id

        self.best_val = float("inf")

        self.save_hyperparameters("optim_str", "learning_rate", "batch_size", "exp_id")
    
    def on_fit_start(self):
        dm = self.trainer.datamodule
        self.drift_model = DriftModel(
            self.drift_model.data_driven_model, dm.stress_normalization, dm.wind_normalization
        )
        self.u_x_normalization = dm.u_x_normalization
        self.u_y_normalization = dm.u_y_normalization
        
    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        buffer = io.BytesIO(checkpoint["drift_model"])
        self.drift_model = eqx.tree_deserialise_leaves(buffer, self.drift_model)
        buffer.close()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        buffer = io.BytesIO()
        eqx.tree_serialise_leaves(buffer, self.drift_model)
        checkpoint["drift_model"] = buffer.getvalue()
        buffer.close()

    def configure_optimizers(self) -> None:
        optim = optax.adam
        if self.optim_str == "adamw":
            optim = optax.adamw

        self.optim = optax.chain(optax.clip_by_global_norm(1.0), optim(learning_rate=self.learning_rate))
        self.opt_state = self.optim.init(eqx.filter(self.drift_model, eqx.is_array))

        return None

    def training_step(self, batch: Float[np.ndarray, "batch_size n_features"]) -> torch.Tensor:
        batch = self._to_jax(batch)

        self.drift_model, self.opt_state, loss = self.make_training_step(
            self.optim, self.drift_model, self.opt_state, batch
        )

        loss = torch.from_numpy(np.asarray(loss))
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: Float[np.ndarray, "batch_size n_features"]) -> torch.Tensor:
        batch = self._to_jax(batch)

        val = self.make_validation_step(self.drift_model, batch) 

        val = torch.from_numpy(np.asarray(val))
        self.log("val", val, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if val < self.best_val:
            self.best_val = val
            self.trainer.save_checkpoint("best_model.ckpt")

        return val

    @classmethod
    @eqx.filter_jit
    def make_training_step(
        cls, 
        optim: optax.GradientTransformation,
        drift_model: DriftModel,
        opt_state: optax.OptState, 
        batch: Float[jax.Array, "batch_size n_features"], 
    ) -> tuple[DriftModel, optax.OptState, Float[jax.Array, ""]]:
        loss, grads = eqx.filter_value_and_grad(cls._loss_fn)(drift_model, batch)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(drift_model, eqx.is_array))
        drift_model = eqx.apply_updates(drift_model, updates)

        return drift_model, opt_state, loss

    @classmethod
    @eqx.filter_jit
    def make_validation_step(
        cls, drift_model: DriftModel, batch: Float[jax.Array, "batch_size n_features"], 
    ) -> Float[jax.Array, ""]:
        return cls._loss_fn(drift_model, batch)
    
    @classmethod
    def _loss_fn(
        cls, drift_model: DriftModel, batch: Float[jax.Array, "batch_size n_features"]
    ) -> Float[jax.Array, ""]:
        log_likelihoods = cls._log_likelihoods_fn(drift_model, batch)
        return -jnp.mean(log_likelihoods)
    
    @classmethod
    def _log_likelihoods_fn(
        cls, drift_model: DriftModel, batch: Float[jax.Array, "batch_size n_features"]
    ) -> Float[jax.Array, "batch_size"]:
        (
            ve, vn, 
            ugos, vgos, 
            eastward_stress, northward_stress, 
            eastward_wind, northward_wind, 
            moy_cos, moy_sin,
            lat_cos, lat_sin,
            lon_cos, lon_sin
        ) = cls._decompose_batch(batch)

        log_likelihoods = jax.vmap(drift_model.compute_log_likelihood_from_features)(
            ve, vn,
            ugos, vgos,
            eastward_stress, northward_stress, 
            eastward_wind, northward_wind, 
            moy_cos, moy_sin,
            lat_cos, lat_sin,
            lon_cos, lon_sin
        )
        
        return log_likelihoods
    
    @classmethod
    def _decompose_batch(
        cls, batch: Float[jax.Array, "batch_size n_features"]
    ) -> tuple[
        Float[jax.Array, "batch_size"], Float[jax.Array, "batch_size"], 
        Float[jax.Array, "batch_size"], Float[jax.Array, "batch_size"], 
        Float[jax.Array, "batch_size"], Float[jax.Array, "batch_size"], 
        Float[jax.Array, "batch_size"], Float[jax.Array, "batch_size"], 
        Float[jax.Array, "batch_size"], Float[jax.Array, "batch_size"], 
        Float[jax.Array, "batch_size"], Float[jax.Array, "batch_size"], 
        Float[jax.Array, "batch_size"], Float[jax.Array, "batch_size"]
    ]:
        ve, vn = batch[:, 3], batch[:, 4]
        ugos, vgos = batch[:, 5], batch[:, 6]
        eastward_stress, northward_stress = batch[:, 7], batch[:, 8]
        eastward_wind, northward_wind = batch[:, 9], batch[:, 10]
        moy_cos, moy_sin = batch[:, 11], batch[:, 12]
        lat_cos, lat_sin = batch[:, 13], batch[:, 14]
        lon_cos, lon_sin = batch[:, 15], batch[:, 16]
        return (
            ve, vn, 
            ugos, vgos, 
            eastward_stress, northward_stress, 
            eastward_wind, northward_wind, 
            moy_cos, moy_sin,
            lat_cos, lat_sin,
            lon_cos, lon_sin
        )
    
    @classmethod
    def _to_jax(cls, batch: Float[np.ndarray, "..."]) -> Float[jax.Array, "..."]:
        return jnp.asarray(batch)
