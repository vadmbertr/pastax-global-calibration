from typing import Callable

import lightning.pytorch as L
from lightning.pytorch.callbacks import Callback


class OnTrainEndCallback(Callback):
    def __init__(self, callback_fn: Callable[[L.Trainer, L.LightningModule], None]):
        super().__init__()
        self.callback_fn = callback_fn

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.callback_fn(trainer, pl_module)
