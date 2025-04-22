import io
from typing import Any

import equinox as eqx
from jax.dlpack import from_dlpack as jax_from_dlpack
import jax.numpy as jnp
import jaxlib
from jaxtyping import Array, Float, Int, Real
import lightning as L
import optax
import torch
from torch.utils.dlpack import to_dlpack as torch_to_dlpack

from pastax.gridded import Gridded
from pastax.simulator import DeterministicSimulator
from pastax.trajectory import Location, Trajectory


class DeterministicModule(L.LightningModule):
    def __init__(
        self, 
        dynamics: eqx.Module,
        integration_horizon: float,  # in days
        integration_dt: float,  # in seconds
        learning_rate: float,
    ):
        super().__init__()
        
        self.automatic_optimization = False
        self.save_hyperparameters("integration_horizon", "integration_dt", "learning_rate")

        self.dynamics = dynamics

        self.integration_horizon = integration_horizon
        self.integration_dt = integration_dt
        self.n_steps = int(self.integration_horizon * 24 * 60 * 60 // self.integration_dt)

        self.learning_rate = learning_rate

        self.simulator = None

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        buffer = io.BytesIO(checkpoint["dynamics"])
        self.dynamics = eqx.tree_deserialise_leaves(buffer, self.dynamics)
        buffer.close()
        self.simulator = DeterministicSimulator()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        buffer = io.BytesIO()
        eqx.tree_serialise_leaves(buffer, self.dynamics)
        checkpoint["dynamics"] = buffer.getvalue()
        buffer.close()

    def setup(self, stage: str):
        if self.simulator is None:
            self.simulator = DeterministicSimulator()

    def configure_optimizers(self):
        self.optim = optax.chain(optax.zero_nans(), optax.adam(self.learning_rate))
        self.opt_state = self.optim.init(eqx.filter(self.dynamics, eqx.is_array))

    def training_step(self, batch: tuple[list[torch.Tensor], list[torch.Tensor]]) -> torch.Tensor:
        reference_trajectory_batch, grid_batch = self._tensors_to_pytrees(*batch)

        self.dynamics, self.opt_state, loss, grad = self.make_step(
            self.optim, self.simulator, self.integration_dt, self.n_steps, 
            self.dynamics, self.opt_state, grid_batch, reference_trajectory_batch
        )

        loss = torch.scalar_tensor(loss.item())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for name, param in grad.get_parameters().items():
            self.log(
                f"grad_{name}", torch.scalar_tensor(param.item()), 
                on_step=True, on_epoch=True, logger=True
            )

        for name, param in self.dynamics.get_parameters().items():
            self.log(
                f"param_{name}", torch.scalar_tensor(param.item()), 
                on_step=True, on_epoch=True, logger=True
            )

        return loss

    def validation_step(self, batch: tuple[list[torch.Tensor], list[torch.Tensor]]) -> torch.Tensor:
        reference_trajectory_batch, grid_batch = self._tensors_to_pytrees(*batch)

        loss, _ = self._batch_loss_fn(
            self.dynamics, self.simulator, self.integration_dt, self.n_steps,
            grid_batch, reference_trajectory_batch
        ) 

        loss = torch.scalar_tensor(loss.item())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def forward(self, grid: Gridded, x0: Location, ts: Real[Array, "time"]) -> Trajectory:
        return self._forward(
            self.simulator, self.integration_dt, self.n_steps, self.dynamics, grid, x0, ts
        )
    
    @classmethod
    def _tensors_to_pytrees(
        cls,
        reference_trajectory_batch: list[torch.Tensor],
        grid_batch: list[torch.Tensor]
    ) -> tuple[Trajectory, Gridded]:
        reference_trajectory_batch = [cls._torch_to_jax(arr) for arr in reference_trajectory_batch]
        grid_batch = [cls._torch_to_jax(arr) for arr in grid_batch]
        
        reference_trajectory_batch = cls._to_trajectory(reference_trajectory_batch)
        grid_batch = cls._to_gridded(grid_batch)

        return reference_trajectory_batch, grid_batch
    
    @classmethod
    def _torch_to_jax(cls, array: torch.Tensor) -> Float[Array, "..."]:
        to_jax = lambda arr: jax_from_dlpack(torch_to_dlpack(arr), copy=False)

        array = array.detach()
        try:
            array = to_jax(array)
        except jaxlib.xla_extension.XlaRuntimeError:  # need to flatten first (copy might happen...)
            shape = array.shape
            array = to_jax(array.flatten()).reshape(shape)
        return array

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
            Float[Array, "time lat lon"], 
            Float[Array, "time lat lon"], 
            Float[Array, "time "],
            Float[Array, "lat"], 
            Float[Array, "lon"]
        ]
    ) -> Gridded:
        u, v, time, lat, lon = field_arrays
        
        gridded = eqx.filter_vmap(Gridded.from_array)(
            {"u": u, "v": v}, time, lat, lon
        )

        return gridded

    @classmethod
    @eqx.filter_jit
    def make_step(
        cls, 
        optim: optax.GradientTransformation,
        simulator: DeterministicSimulator, 
        integration_dt: float, 
        n_steps: int, 
        dynamics: eqx.Module,
        opt_state: optax.OptState, 
        grid_batch: Gridded, 
        reference_trajectory_batch: Trajectory
    ):
        grad, loss_val = eqx.filter_jacfwd(cls._batch_loss_fn, has_aux=True)(
            dynamics, simulator, integration_dt, n_steps, grid_batch, reference_trajectory_batch
        )
        updates, opt_state = optim.update(grad, opt_state)
        dynamics = eqx.apply_updates(dynamics, updates)
        return dynamics, opt_state, loss_val, grad

    @classmethod
    def _batch_loss_fn(
        cls, 
        dynamics: eqx.Module,
        simulator: DeterministicSimulator, 
        integration_dt: float, 
        n_steps: int, 
        grid_batch: Gridded, 
        reference_trajectory_batch: Trajectory
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        loss = eqx.filter_vmap(
            lambda grid, reference_trajectory: cls._sample_loss_fn(
                dynamics, simulator, integration_dt, n_steps, grid, reference_trajectory
            )
        )(grid_batch, reference_trajectory_batch).mean()
        return loss, loss

    @classmethod
    def _sample_loss_fn(
        cls, 
        dynamics: eqx.Module,
        simulator: DeterministicSimulator, 
        integration_dt: float, 
        n_steps: int, 
        grid: Gridded, 
        reference_trajectory: Trajectory
    ) -> Float[Array, ""]:
        x0 = reference_trajectory.origin
        ts = reference_trajectory.times.value
        simulated_trajectory = cls._forward(simulator, integration_dt, n_steps, dynamics, grid, x0, ts)

        residuals = reference_trajectory.liu_index(simulated_trajectory).value
        return (residuals ** 2).sum()
    
    @classmethod
    def _forward(
        cls, 
        simulator: DeterministicSimulator, 
        integration_dt: float, 
        n_steps: int, 
        dynamics: eqx.Module,
        grid: Gridded, 
        x0: Location, 
        ts: Real[Array, "time"]
    ) -> Trajectory:
        dt0, saveat, stepsize_controller, adjoint, n_steps, _ = simulator.get_diffeqsolve_best_args(
            ts, integration_dt, n_steps=n_steps, constant_step_size=True, save_at_steps=False, ad_mode="forward"
        )

        return simulator(
            dynamics=dynamics, args=grid, x0=x0, ts=ts, 
            dt0=dt0, saveat=saveat, stepsize_controller=stepsize_controller, adjoint=adjoint, max_steps=n_steps
        )
