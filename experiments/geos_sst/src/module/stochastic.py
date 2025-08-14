import io
from typing import Any, Callable

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxlib
from jaxtyping import Array, Float, Int, Real
import lightning as L
import optax
import torch

from pastax.gridded import Gridded
from pastax.simulator import StochasticSimulator
from pastax.trajectory import Location, Trajectory, TrajectoryEnsemble


class StochasticModule(L.LightningModule):
    def __init__(
        self, 
        dynamics: eqx.Module,
        ensemble_size: int,
        integration_horizon: float,  # in days
        integration_dt: float,  # in seconds
        optimizer: str,
        learning_rate_scheduler: str,
        learning_rate: float,
        loss_fn: str
    ):
        super().__init__()
        
        self.automatic_optimization = False
        self.save_hyperparameters(
            "ensemble_size", 
            "integration_horizon", "integration_dt", 
            "optimizer", "learning_rate_scheduler", "learning_rate"
        )

        self.simulator = StochasticSimulator()

        self.dynamics = dynamics
        self.ensemble_size = ensemble_size

        self.integration_horizon = integration_horizon
        self.integration_dt = integration_dt
        self.n_steps = int(self.integration_horizon * 24 * 60 * 60 // self.integration_dt)

        self.optimizer = optimizer
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate = learning_rate

        self.loss_fn = loss_fn

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        buffer = io.BytesIO(checkpoint["dynamics"])
        self.dynamics = eqx.tree_deserialise_leaves(buffer, self.dynamics)
        buffer.close()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        buffer = io.BytesIO()
        eqx.tree_serialise_leaves(buffer, self.dynamics)
        checkpoint["dynamics"] = buffer.getvalue()
        buffer.close()

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

        self.optim = optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(1.0),
            optim(lr_scheduler(self.learning_rate))
        )
        self.opt_state = self.optim.init(eqx.filter(self.dynamics, eqx.is_inexact_array))

    def training_step(self, batch: tuple[list[torch.Tensor], list[torch.Tensor]]) -> torch.Tensor:
        traj_len = batch[0][0].shape[1]
        
        reference_trajectory_batch, grids_batch = self._tensors_to_pytrees(*batch)
        indices = self._get_loss_indices(traj_len)

        self.dynamics, self.opt_state, loss, grads, metrics = self.make_step(
            self.simulator,
            self.dynamics,
            self.ensemble_size,
            self.integration_dt,
            self.n_steps,
            self.loss_fn,
            self.optim,
            grids_batch,
            reference_trajectory_batch,
            indices
        )

        loss = self._jax_to_torch(loss)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({f"train_{k}": v.item() for k, v in metrics.items()}, on_step=True, on_epoch=True, logger=True)
 
        return loss

        all_finite = True
        for model_gradients in grads.get_gradients():
            if not jnp.all(jnp.isfinite(model_gradients)):
                all_finite = False

        if not all_finite:
            print("not all gradients are finite")

            simulated_trajectories_batch = eqx.filter_vmap(
                lambda grids, random_variables, reference_trajectory: self.forward(
                    grids, random_variables, reference_trajectory.origin, reference_trajectory.times.value
                )
            )(grids_batch, random_variables_batch, reference_trajectory_batch)

            debug_loss = eqx.filter_vmap(
                lambda reference_trajectory, simulated_trajectories: self._separation_distance(
                    reference_trajectory, simulated_trajectories, indices
                )
            )(reference_trajectory_batch, simulated_trajectories_batch)

            highest_loss = jnp.argmax(debug_loss).item()

            print(f"mean loss: {debug_loss.mean()}")
            print(f"highest loss: {debug_loss[highest_loss]}")
            print(f"ref origin: {reference_trajectory_batch.locations.value[highest_loss, 0]}")
            print(f"ref end: {reference_trajectory_batch.locations.value[highest_loss, -1]}")
            print(f"sim end min: {simulated_trajectories_batch.locations.value[highest_loss, :, -1].min(axis=0)}")
            print(f"sim end mean: {simulated_trajectories_batch.locations.value[highest_loss, :, -1].mean(axis=0)}")
            print(f"sim end max: {simulated_trajectories_batch.locations.value[highest_loss, :, -1].max(axis=0)}")

            def _steps_velocities_ensemble(
                traj_ens: TrajectoryEnsemble
            ) -> tuple[Float[Array, "traj_len"], Float[Array, "traj_len"]]:
                def _velocities(traj: Trajectory) -> Float[Array, "traj_len"]:
                    steps = traj.steps().value

                    times = traj.times.value[1:] - traj.times.value[:-1]
                    times = jnp.pad(times, (1, 0), constant_values=1e-4)

                    return steps / times
                return traj_ens.steps().value, traj_ens.map(_velocities).value
            
            steps_ensemble, velocities_ensemble = eqx.filter_vmap(
                _steps_velocities_ensemble
            )(simulated_trajectories_batch)

            print(f"max sim step: {steps_ensemble[highest_loss].max()}")
            print(f"max sim velocity: {velocities_ensemble[highest_loss].max()}")

            print(f"max ug velocity: {batch[1][0][0][highest_loss].max()}")
            print(f"max vg velocity: {batch[1][0][1][highest_loss].max()}")
            print(f"max uw velocity: {batch[1][1][0][highest_loss].max()}")
            print(f"max vw velocity: {batch[1][1][1][highest_loss].max()}")
            print(f"max us velocity: {batch[1][2][0][highest_loss].max()}")
            print(f"max vs velocity: {batch[1][2][1][highest_loss].max()}")
            print(f"max ts velocity: {batch[1][2][2][highest_loss].max()}")

            jax.debug.breakpoint()

        return loss

    def validation_step(self, batch: tuple[list[torch.Tensor], list[torch.Tensor]]) -> torch.Tensor:
        traj_len = batch[0][0].shape[1]

        reference_trajectory_batch, grids_batch = self._tensors_to_pytrees(*batch)
        indices = self._get_loss_indices(traj_len)

        _, (loss, metrics) = self._batch_step(
            self.simulator,
            self.dynamics,
            self.ensemble_size,
            self.integration_dt,
            self.n_steps,
            self.loss_fn,
            grids_batch,
            reference_trajectory_batch,
            indices
        )

        loss = self._jax_to_torch(loss)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({f"val_{k}": v.item() for k, v in metrics.items()}, on_step=True, on_epoch=True, logger=True)
        
        return loss

    def forward(self, grids: Gridded, x0: Location, ts: Real[Array, "time"]) -> TrajectoryEnsemble:
        return self._forward(
            self.simulator, self.integration_dt, self.n_steps, self.dynamics, grids, x0, ts
        )

    @classmethod
    @eqx.filter_jit
    def make_step(
        cls, 
        simulator: StochasticSimulator,
        dynamics: eqx.Module,
        ensemble_size: int,
        integration_dt: float,
        n_steps: int,
        loss_fn: str,
        optim: optax.GradientTransformation,
        grids_batch: tuple[Gridded, Gridded, Gridded], 
        reference_trajectory_batch: Trajectory,
        indices: Int[Array, ""]
    ) -> tuple[eqx.Module, optax.OptState, Float[Array, ""], eqx.Module, dict[str, Float[Array, ""]]]:
        grad, (loss_val, metrics) = eqx.filter_jacfwd(cls._batch_step, has_aux=True)(
            simulator,
            dynamics,
            ensemble_size,
            integration_dt,
            n_steps,
            loss_fn,
            grids_batch,
            reference_trajectory_batch,
            indices
        )

        grad = eqx.filter(grad, eqx.is_array)
        updates, opt_state = optim.update(grad, opt_state, dynamics)
        dynamics = eqx.apply_updates(dynamics, updates)
        
        return dynamics, opt_state, loss_val, grad, metrics

    @classmethod
    def _batch_step(
        cls,
        simulator: StochasticSimulator,
        dynamics: eqx.Module,
        ensemble_size: int,
        integration_dt: float,
        n_steps: int,
        loss_fn: str,
        grids_batch: tuple[Gridded, Gridded, Gridded], 
        reference_trajectory_batch: Trajectory,
        indices: Int[Array, ""]
    ) -> tuple[Float[Array, ""], tuple[Float[Array, ""], dict[str, Float[Array, ""]]]]:
        loss, metrics = eqx.filter_vmap(
            lambda grid, reference_trajectory: cls._sample_step(
                simulator,
                dynamics,
                ensemble_size,
                integration_dt,
                n_steps,
                loss_fn,
                grid,
                reference_trajectory,
                indices
            )
        )(grids_batch, reference_trajectory_batch)

        loss = loss.mean()
        metrics = {k: v.mean() for k, v in metrics.items()}

        return loss, (loss, metrics)

    @classmethod
    def _sample_step(
        cls,
        simulator: StochasticSimulator,
        dynamics: eqx.Module,
        ensemble_size: int,
        integration_dt: float,
        n_steps: int,
        loss_fn: str,
        grids: tuple[Gridded, Gridded, Gridded],  
        reference_trajectory: Trajectory,
        indices: Int[Array, ""]
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        x0 = reference_trajectory.origin
        ts = reference_trajectory.times.value
        simulated_trajectories = cls._forward(
            simulator,
            dynamics,
            ensemble_size,
            integration_dt,
            n_steps,
            grids,
            x0,
            ts
        )

        liu_index = cls._liu_index(reference_trajectory, simulated_trajectories, indices)
        separation_distance = cls._separation_distance(reference_trajectory, simulated_trajectories, indices)

        if loss_fn == "liu_index":
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
        simulator: StochasticSimulator,
        dynamics: eqx.Module,
        ensemble_size: int,
        integration_dt: float,
        n_steps: int,
        grids: tuple[Gridded, Gridded],
        x0: Location,
        ts: Real[Array, "time"]
    ) -> TrajectoryEnsemble:
        dt0, saveat, stepsize_controller, adjoint, n_steps, brownian_motion = simulator.get_diffeqsolve_best_args(
            ts,
            integration_dt,
            n_steps,
            constant_step_size=True,
            save_at_steps=False,
            ad_mode="reverse"
        )

        simulated_trajectories = simulator(
            dynamics=dynamics,
            args=grids,
            x0=x0,
            ts=ts,
            solver=dfx.Heun(),
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            max_steps=n_steps,
            n_samples=ensemble_size,
            brownian_motion=brownian_motion
        )

        return simulated_trajectories
    
    @classmethod
    def _liu_index(
        cls,
        reference_trajectory: Trajectory,
        simulated_trajectories: TrajectoryEnsemble,
        indices: Int[Array, ""]
    ) -> Float[Array, ""]:
        pair_pair_residuals_fn = lambda traj1, traj2: traj1.liu_index(traj2).value

        return cls._crps(reference_trajectory, simulated_trajectories, pair_pair_residuals_fn, indices)

    @classmethod
    def _separation_distance(
        cls,
        reference_trajectory: Trajectory,
        simulated_trajectories: TrajectoryEnsemble,
        indices: Int[Array, ""]
    ) -> Float[Array, ""]:
        pair_pair_residuals_fn = lambda traj1, traj2: traj1.separation_distance(traj2).value / traj1.lengths().value

        return cls._crps(reference_trajectory, simulated_trajectories, pair_pair_residuals_fn, indices)

    @classmethod
    def _crps(
        cls,
        reference_trajectory: Trajectory,
        simulated_trajectories: TrajectoryEnsemble,
        pair_pair_residuals_fn: Callable,
        indices: Int[Array, ""]
    ) -> Float[Array, ""]:
        residuals = simulated_trajectories.crps(
            reference_trajectory, pair_pair_residuals_fn, is_metric_symmetric=False
        ).value

        residuals = residuals[indices]

        return residuals.mean()

    @classmethod
    def _get_loss_indices(cls, traj_len: int) -> Int[Array, ""]:  # puts more weight on the beginning of the trajectory
        exp_space = jnp.logspace(0, 1, traj_len // 3, base=10, endpoint=False) - 1
        exp_space = exp_space / exp_space[-1]
        indices = jnp.round(exp_space * (traj_len - 2)).astype(int) + 1
        return indices

    def _get_random_variables(self, batch_size: int) -> Float[Array, "batch_size ensemble_size n_steps n_coeffs"]:
        if self.antithetic_variate:
            random_variables = jax.random.normal(
                jax.random.PRNGKey(0), (batch_size, self.ensemble_size // 2, self.n_steps, self.dynamics.mu.size)
            )
            random_variables = jnp.concatenate((random_variables, -random_variables), axis=1)
        else:
            random_variables = jax.random.normal(
                jax.random.PRNGKey(0), (batch_size, self.ensemble_size, self.n_steps, self.dynamics.mu.size)
            )
        
        return random_variables

    @classmethod
    def _jax_to_torch(cls, array: Float[Array, "..."]) -> torch.Tensor:
        try:
            array = torch.from_dlpack(array)
        except jaxlib.xla_extension.XlaRuntimeError:  # need to flatten first (copy might happen...)
            shape = array.shape
            array = torch.from_dlpack(array.ravel()).reshape(shape)
        return array
    
    @classmethod
    def _torch_to_jax(cls, array: torch.Tensor) -> Float[Array, "..."]:
        array = array.detach()
        try:
            array = jnp.from_dlpack(array, copy=None)
        except jaxlib.xla_extension.XlaRuntimeError:  # need to flatten first (copy might happen...)
            shape = array.shape
            array = jnp.from_dlpack(array.ravel(), copy=None).reshape(shape)
        return array

    @classmethod
    def _tensors_to_pytrees(
        cls,
        reference_trajectory_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        grid_batch: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
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
                Float[Array, "time "],
                Float[Array, "lat"], 
                Float[Array, "lon"]
            ]
        ]
    ) -> tuple[Gridded, Gridded]:
        def _to_gridded(variables, arrays):
            return eqx.filter_vmap(Gridded.from_array)(
                {variables[i]: arrays[i] for i in range(len(variables))}, *arrays[-3:]
            )
    
        duacs_arr, mur_arr = field_arrays
        duacs_ds = _to_gridded(("u", "v"), duacs_arr)
        mur_ds = _to_gridded(("T"), mur_arr)

        return duacs_ds, mur_ds
