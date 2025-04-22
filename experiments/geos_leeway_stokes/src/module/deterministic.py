import io
from typing import Any

import diffrax as dfx
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
        integration_dt: float,  # in seconds
        learning_rate: float,
    ):
        super().__init__()
        
        self.automatic_optimization = False
        self.save_hyperparameters("integration_dt", "learning_rate")

        self.dynamics = dynamics
        self.fixed_dynamics = dynamics

        self.integration_horizon = 5  # days
        self.integration_dt = integration_dt
        self.n_steps = int(self.integration_horizon * 24 * 60 * 60 // self.integration_dt)

        self.learning_rate = learning_rate

        self.simulator = DeterministicSimulator()

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
        self.optim = optax.chain(
            optax.zero_nans(), 
            optax.adam(self.learning_rate, b1=.6),
            optax.clip(1),
            optax.keep_params_nonnegative(),
        )
        self.opt_state = self.optim.init(eqx.filter(self.dynamics, eqx.is_inexact_array))

    def training_step(self, batch: tuple[list[torch.Tensor], list[torch.Tensor]]) -> torch.Tensor:
        traj_len = batch[0][0].shape[1]
        reference_trajectory_batch, grid_batch = self._tensors_to_pytrees(*batch)

        self.dynamics, self.opt_state, loss, grad = self.make_step(
            self.optim, self.simulator, self.integration_dt, self.n_steps, 
            self.dynamics, self.opt_state, grid_batch, reference_trajectory_batch, traj_len
        )

        loss = torch.scalar_tensor(loss.item())

        print(f"loss: {loss.item()}")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for name, param in grad.get_parameters().items():
            print(f"grad_{name}: {param.item()}")
            try:
                self.log(
                    f"grad_{name}", torch.scalar_tensor(param.item()), 
                    on_step=True, on_epoch=True, logger=True
                )
            except:
                print(f"grad_{name} failed to log")

        for name, param in self.dynamics.get_parameters().items():
            print(f"param_{name}: {param.item()}")
            self.log(
                f"param_{name}", torch.scalar_tensor(param.item()), 
                on_step=True, on_epoch=True, logger=True
            )

        print(f"ref origin: {reference_trajectory_batch.locations.value[0][0]}")
        print(f"ref end: {reference_trajectory_batch.locations.value[0][-1]}")

        trajs = eqx.filter_vmap(
            lambda grid, reference_trajectory: self.forward(
                grid, reference_trajectory.origin, reference_trajectory.times.value
            )
        )(grid_batch, reference_trajectory_batch)

        print(f"tuned end: {trajs.locations.value[0][-1]}")

        trajs = eqx.filter_vmap(
            lambda grid, reference_trajectory: self._forward(
                self.simulator, self.integration_dt, self.n_steps, self.fixed_dynamics, 
                grid, reference_trajectory.origin, reference_trajectory.times.value
            )
        )(grid_batch, reference_trajectory_batch)

        print(f"fixed end: {trajs.locations.value[0][-1]}")

        return loss

    def validation_step(self, batch: tuple[list[torch.Tensor], list[torch.Tensor]]) -> torch.Tensor:
        traj_len = batch[0][0].shape[1]
        reference_trajectory_batch, grid_batch = self._tensors_to_pytrees(*batch)

        loss, _ = self._batch_loss_fn(
            self.dynamics, self.simulator, self.integration_dt, self.n_steps,
            grid_batch, reference_trajectory_batch, traj_len
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
        reference_trajectory_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        grid_batch: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ]
    ) -> tuple[Trajectory, Gridded]:
        reference_trajectory_batch = [cls._torch_to_jax(arr) for arr in reference_trajectory_batch]
        grid_batch = [[cls._torch_to_jax(arr) for arr in grid] for grid in grid_batch]
        
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
                Float[Array, "time "],
                Float[Array, "lat"], 
                Float[Array, "lon"]
            ]
        ]
    ) -> tuple[Gridded, Gridded, Gridded]:
        def _to_gridded(u, v, time, lat, lon):
            return eqx.filter_vmap(Gridded.from_array)(
                {"u": u, "v": v}, time, lat, lon
            )
    
        uc, uw, uh = field_arrays
        uc = _to_gridded(*uc)
        uw = _to_gridded(*uw)
        uh = _to_gridded(*uh)

        return uc, uw, uh

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
        grid_batch: tuple[Gridded, Gridded, Gridded], 
        reference_trajectory_batch: Trajectory,
        traj_len: int | None = None
    ):
        grad, loss_val = eqx.filter_jacfwd(cls._batch_loss_fn, has_aux=True)(
            dynamics, simulator, integration_dt, n_steps, grid_batch, reference_trajectory_batch, traj_len
        )

        grad = eqx.filter(grad, eqx.is_array)
        grad = eqx.tree_at(lambda t: t.drag, grad, replace_fn=lambda n: n * .1)
        grad = eqx.tree_at(lambda t: t.wave, grad, replace_fn=lambda n: n * 10)
        updates, opt_state = optim.update(grad, opt_state, dynamics)
        dynamics = eqx.apply_updates(dynamics, updates)
        
        return dynamics, opt_state, loss_val, grad

    @classmethod
    def _batch_loss_fn(
        cls, 
        dynamics: eqx.Module,
        simulator: DeterministicSimulator, 
        integration_dt: float, 
        n_steps: int, 
        grid_batch: tuple[Gridded, Gridded, Gridded], 
        reference_trajectory_batch: Trajectory,
        traj_len: int | None
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        loss = eqx.filter_vmap(
            lambda grid, reference_trajectory: cls._sample_loss_fn(
                dynamics, simulator, integration_dt, n_steps, grid, reference_trajectory, traj_len
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
        grid: tuple[Gridded, Gridded, Gridded], 
        reference_trajectory: Trajectory,
        traj_len: int | None
    ) -> Float[Array, ""]:
        x0 = reference_trajectory.origin
        ts = reference_trajectory.times.value
        simulated_trajectory = cls._forward(simulator, integration_dt, n_steps, dynamics, grid, x0, ts)

        residuals = reference_trajectory.liu_index(simulated_trajectory).value

        if traj_len is not None:
            exp_space = jnp.logspace(0, 1, traj_len // 3, base=10, endpoint=False) - 1
            exp_space = exp_space / exp_space[-1]
            indices = jnp.round(exp_space * (traj_len - 1)).astype(int)
            residuals = residuals[indices]

        return (residuals ** 2).sum()
    
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
