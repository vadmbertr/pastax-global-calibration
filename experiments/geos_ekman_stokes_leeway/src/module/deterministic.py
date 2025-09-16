import os

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Real
import lightning as L
import optax
import torch

from pastax.gridded import Gridded
from pastax.simulator import DeterministicSimulator
from pastax.trajectory import Location, Trajectory


class DeterministicModule(L.LightningModule):
    def __init__(
        self, 
        dynamics: eqx.Module,
        integration_dt: float,  # in seconds
        optimizer: str,
        learning_rate_scheduler: str,
        learning_rate: float,
        loss_fn: str,
        default_root_dir: str,
    ):
        super().__init__()
        
        self.automatic_optimization = False
        self.save_hyperparameters("integration_dt", "learning_rate_scheduler", "optimizer", "learning_rate", "loss_fn")
        
        self.checkpoints_dir = f"{default_root_dir}/checkpoints" if default_root_dir else "checkpoints"

        self.simulator = DeterministicSimulator()

        self.dynamics = dynamics

        self.integration_horizon = 5  # days
        self.integration_dt = integration_dt
        self.n_steps = int(self.integration_horizon * 24 * 60 * 60 // self.integration_dt)

        self.optimizer = optimizer
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate = learning_rate

        self.loss_fn = loss_fn

        self.min_val_loss = float("inf")

    def on_fit_start(self):
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        try:
            self._reload_best_dynamics()
        except:
            pass

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"]
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            eqx.tree_serialise_leaves(f"{self.checkpoints_dir}/best.eqx", self.dynamics)

    def on_fit_end(self):
        eqx.tree_serialise_leaves(f"{self.checkpoints_dir}/last.eqx", self.dynamics)

    def configure_optimizers(self):
        optim = optax.adam
        if self.optimizer == "adamW":
            optim = optax.adamw
        elif self.optimizer == "adabelief":
            optim = optax.adabelief
        elif self.optimizer == "rmsprop":
            optim = optax.rmsprop

        lr_scheduler = optax.constant_schedule
        if self.learning_rate_scheduler == "cosine":
            lr_scheduler = lambda lr: optax.cosine_onecycle_schedule(
                transition_steps=1000,
                peak_value=lr*2.5,
                pct_start=0.1,
                div_factor=2.5,
                final_div_factor=10.0
            )

        optim = optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(1.0),
            optim(lr_scheduler(self.learning_rate))
        )
        self.optim = optax.apply_if_finite(optim, max_consecutive_errors=100)
        self.opt_state = self.optim.init(eqx.filter(self.dynamics, eqx.is_inexact_array))

    def training_step(self, batch: tuple[list[torch.Tensor], list[torch.Tensor]]) -> torch.Tensor:
        traj_len = batch[0][0].shape[1]

        reference_trajectory_batch, grid_batch = self._tensors_to_pytrees(*batch)
        indices = self._get_loss_indices(traj_len)

        self.dynamics, self.opt_state, loss, grads, metrics = self.make_step(
            self.optim, self.simulator, self.integration_dt, self.n_steps, 
            self.dynamics, self.opt_state, grid_batch, reference_trajectory_batch, indices
        )

        loss = self._jax_to_torch(loss)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({f"train_{k}": v.item() for k, v in metrics.items()}, on_step=True, on_epoch=True, logger=True)

        ekman_scale, ekman_rotation, stokes_scale, leeway_scale = self.dynamics.get_coefficients()

        self.log_dict(
            {
                "coef_ekman_scale": ekman_scale.item(),
                "coef_ekman_rotation": jnp.rad2deg(ekman_rotation).item(),
                "coef_stokes_scale": stokes_scale.item(),
                "coef_leeway_scale": leeway_scale.item()
            }, 
            on_step=False, 
            on_epoch=True, 
            logger=True
        )

        return loss

    def validation_step(self, batch: tuple[list[torch.Tensor], list[torch.Tensor]]) -> torch.Tensor:
        traj_len = batch[0][0].shape[1]

        reference_trajectory_batch, grid_batch = self._tensors_to_pytrees(*batch)
        indices = self._get_loss_indices(traj_len)

        _, (loss, metrics) = self._batch_step(
            self.dynamics, self.simulator, self.integration_dt, self.n_steps,
            grid_batch, reference_trajectory_batch, indices
        )

        loss = self._jax_to_torch(loss)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({f"val_{k}": v.item() for k, v in metrics.items()}, on_step=False, on_epoch=True, logger=True)
        
        return loss

    def forward(self, grid: Gridded, x0: Location, ts: Real[Array, "time"]) -> Trajectory:
        return self._forward(
            self.simulator, self.integration_dt, self.n_steps, self.dynamics, grid, x0, ts
        )

    @eqx.filter_jit
    def make_step(
        self, 
        optim: optax.GradientTransformation,
        simulator: DeterministicSimulator, 
        integration_dt: float, 
        n_steps: int, 
        dynamics: eqx.Module,
        opt_state: optax.OptState, 
        grid_batch: tuple[Gridded, Gridded, Gridded], 
        reference_trajectory_batch: Trajectory,
        indices: Int[Array, ""]
    ) -> tuple[eqx.Module, optax.OptState, Float[Array, ""], eqx.Module, dict[str, Float[Array, ""]]]:
        grad, (loss_val, metrics) = eqx.filter_jacfwd(self._batch_step, has_aux=True)(
            dynamics, simulator, integration_dt, n_steps, grid_batch, reference_trajectory_batch, indices
        )

        grad = eqx.filter(grad, eqx.is_array)
        updates, opt_state = optim.update(grad, opt_state, dynamics)
        dynamics = eqx.apply_updates(dynamics, updates)
        
        return dynamics, opt_state, loss_val, grad, metrics

    def _batch_step(
        self, 
        dynamics: eqx.Module,
        simulator: DeterministicSimulator, 
        integration_dt: float, 
        n_steps: int, 
        grid_batch: tuple[Gridded, Gridded, Gridded], 
        reference_trajectory_batch: Trajectory,
        indices: Int[Array, ""]
    ) -> tuple[Float[Array, ""], tuple[Float[Array, ""], dict[str, Float[Array, ""]]]]:
        loss, metrics = eqx.filter_vmap(
            lambda grid, reference_trajectory: self._sample_step(
                dynamics, simulator, integration_dt, n_steps, grid, reference_trajectory, indices
            )
        )(grid_batch, reference_trajectory_batch)

        loss = loss.mean()
        metrics = {k: v.mean() for k, v in metrics.items()}

        return loss, (loss, metrics)

    def _sample_step(
        self, 
        dynamics: eqx.Module,
        simulator: DeterministicSimulator, 
        integration_dt: float, 
        n_steps: int, 
        grid: tuple[Gridded, Gridded, Gridded], 
        reference_trajectory: Trajectory,
        indices: Int[Array, ""]
    ) -> Float[Array, ""]:
        x0 = reference_trajectory.origin
        ts = reference_trajectory.times.value
        simulated_trajectory = self._forward(simulator, integration_dt, n_steps, dynamics, grid, x0, ts)

        liu_index = self._liu_index(reference_trajectory, simulated_trajectory, indices)
        separation_distance = self._separation_distance(reference_trajectory, simulated_trajectory, indices)

        if self.loss_fn == "liu_index":
            loss = liu_index
        else:
            loss = separation_distance
            
        metrics = {
            "liu_index": liu_index,
            "separation_distance": separation_distance
        }

        return loss, metrics
    
    @classmethod
    def _forward(
        cls, 
        simulator: DeterministicSimulator, 
        integration_dt: float, 
        n_steps: int, 
        dynamics: eqx.Module,
        grid: tuple[Gridded, Gridded, Gridded], 
        x0: Location, 
        ts: Real[Array, "time"]
    ) -> Trajectory:
        dt0, saveat, stepsize_controller, adjoint, n_steps, _ = simulator.get_diffeqsolve_best_args(
            ts, integration_dt, n_steps=n_steps, constant_step_size=True, save_at_steps=False, ad_mode="forward"
        )

        return simulator(
            dynamics=dynamics, args=grid, x0=x0, ts=ts, solver=dfx.Tsit5(),
            dt0=dt0, saveat=saveat, stepsize_controller=stepsize_controller, adjoint=adjoint, max_steps=n_steps
        )

    @classmethod
    def _liu_index(
        cls, reference_trajectory: Trajectory, simulated_trajectory: Trajectory, indices: Int[Array, ""]
    ) -> Float[Array, ""]:
        residuals = reference_trajectory.liu_index(simulated_trajectory).value

        residuals = residuals[indices]

        return residuals.mean()

    @classmethod
    def _separation_distance(
        cls, reference_trajectory: Trajectory, simulated_trajectory: Trajectory, indices: Int[Array, ""]
    ) -> Float[Array, ""]:
        residuals = (
            reference_trajectory.separation_distance(simulated_trajectory).value / reference_trajectory.lengths().value
        )

        residuals = residuals[indices]

        return residuals.mean()

    @classmethod
    def _get_loss_indices(cls, traj_len: int) -> Int[Array, ""]:  # puts more weight on the beginning of the trajectory
        exp_space = jnp.logspace(0, 1, traj_len // 3, base=10, endpoint=False) - 1
        exp_space = exp_space / exp_space[-1]
        indices = jnp.round(exp_space * (traj_len - 2)).astype(int) + 1
        return indices

    @classmethod
    def _jax_to_torch(cls, array: Float[Array, "..."]) -> torch.Tensor:
        try:
            array = torch.from_dlpack(array)
        except:  # need to flatten first (copy might happen...)
            shape = array.shape
            array = torch.from_dlpack(array.ravel()).reshape(shape)
        return array
    
    @classmethod
    def _torch_to_jax(cls, array: torch.Tensor) -> Float[Array, "..."]:
        array = array.detach()
        try:
            array = jnp.from_dlpack(array, copy=None)
        except:  # need to flatten first (copy might happen...)
            shape = array.shape
            array = jnp.from_dlpack(array.ravel(), copy=None).reshape(shape)
        return array

    @classmethod
    def _tensors_to_pytrees(
        cls,
        reference_trajectory_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        grid_batch: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ]
    ) -> tuple[Trajectory, Gridded]:
        reference_trajectory_batch = [cls._torch_to_jax(arr) for arr in reference_trajectory_batch]
        grid_batch = [[cls._torch_to_jax(arr) for arr in grid] for grid in grid_batch]
        
        reference_trajectory_batch = cls._to_trajectory(reference_trajectory_batch)
        grid_batch = cls._to_gridded(grid_batch)

        return reference_trajectory_batch, grid_batch
    
    @classmethod
    @eqx.filter_jit
    def _to_trajectory(
        cls,
        traj_arrays: tuple[Float[Array, "time"], Float[Array, "time"], Float[Array, "time"], Int[Array, ""]]
    ) -> Trajectory:
        traj_lat, traj_lon, traj_time, traj_id = traj_arrays

        traj_latlon = jnp.stack((traj_lat, traj_lon), axis=-1)
        trajectories = eqx.filter_vmap(
            lambda _latlon, _time, _id: Trajectory.from_array(values=_latlon, times=_time, id=_id)
        )(
            traj_latlon, traj_time, traj_id
        )

        return trajectories

    @classmethod
    @eqx.filter_jit
    def _to_gridded(
        cls,
        field_arrays: tuple[
            tuple[
                Float[Array, "time lat lon"], 
                Float[Array, "time lat lon"], 
                Float[Array, "time "],
                Float[Array, "lat"], 
                Float[Array, "lon"]
            ],
            tuple[
                Float[Array, "time lat lon"], 
                Float[Array, "time lat lon"], 
                Float[Array, "time "],
                Float[Array, "lat"], 
                Float[Array, "lon"]
            ],
            tuple[
                Float[Array, "time lat lon"], 
                Float[Array, "time lat lon"], 
                Float[Array, "time lat lon"], 
                Float[Array, "time lat lon"], 
                Float[Array, "time "],
                Float[Array, "lat"], 
                Float[Array, "lon"]
            ]
        ]
    ) -> tuple[Gridded, Gridded, Gridded]:
        def _to_gridded(variables, arrays):
            return eqx.filter_vmap(Gridded.from_array)(
                {variables[i]: arrays[i] for i in range(len(variables))}, *arrays[-3:]
            )
    
        uc, uw, uh = field_arrays
        uc = _to_gridded(("u", "v"), uc)
        uw = _to_gridded(("u", "v"), uw)
        uh = _to_gridded(("u", "v", "t", "h"), uh)

        return uc, uw, uh

    def _reload_best_dynamics(self):
        self.dynamics = eqx.tree_deserialise_leaves(f"{self.checkpoints_dir}/best.eqx", self.dynamics)
