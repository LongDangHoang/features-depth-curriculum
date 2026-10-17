import csv
import math
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from tokenizers import Tokenizer

from . import schedule
from .config import Config
from .mixture import MixtureData
from .module import LanguageModel

LOG_EVERY_N_STEPS = 10
END_OF_TEXT_ID = 0
# Dense bf16 peak per GPU
PEAK_BF16_FLOPS = {"H100": 989e12, "A100": 312e12, "L4": 121e12, "A10G": 125e12, "T4": 65e12}


def is_log_step(steps_done: int) -> bool:
    """Train-side callbacks only compute and log when Lightning flushes, to avoid GPU syncs."""
    return steps_done % LOG_EVERY_N_STEPS == 0


class LossAndGradNorm(L.Callback):
    """Train loss and pre-clip gradient norm."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if is_log_step(trainer.global_step + 1):  # global_step counts the steps already taken
            gradients = [
                parameter.grad for parameter in pl_module.parameters() if parameter.grad is not None
            ]
            pl_module.log("train/grad_norm", torch.nn.utils.get_total_norm(gradients))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_index):
        if is_log_step(trainer.global_step):
            pl_module.log("train/loss", outputs["loss"])


class Throughput(L.Callback):
    """Tokens per second and MFU, timed only on log steps so there is no per-step GPU sync."""

    def __init__(self, flops_per_token: float, peak_flops: float):
        self.flops_per_token = flops_per_token
        self.peak_flops = peak_flops

    def on_train_start(self, trainer, pl_module):
        self.last_time = None  # the first window includes compile warmup, so it is not reported
        self.last_step = trainer.global_step

    def on_validation_end(self, trainer, pl_module):
        # Restart the timer so eval and the checkpoint saved at this step are not counted
        self.last_time = time.perf_counter()
        self.last_step = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_index):
        if not is_log_step(trainer.global_step):
            return
        now = time.perf_counter()
        if self.last_time is not None:
            tokens = (trainer.global_step - self.last_step) * pl_module.config.tokens_per_step
            tokens_per_s = tokens / (now - self.last_time)
            pl_module.log("train/tokens_per_s", tokens_per_s)
            pl_module.log("train/mfu", tokens_per_s * self.flops_per_token / self.peak_flops)
        self.last_time, self.last_step = now, trainer.global_step


class PlannedSchedule(L.Callback):
    """The planned lr, mixture weights and drawn sequences per source of every step, once per run:
    every step as a CSV artifact, and a table and lr plot thinned to W&B's table row limit."""

    def __init__(self, config: Config, datamodule: MixtureData, logger: WandbLogger):
        self.config = config
        self.datamodule = datamodule
        self.logger = logger

    def on_train_start(self, trainer, pl_module):
        config, datamodule = self.config, self.datamodule
        source_ids = datamodule.mixture.source_ids
        columns = ["step", "tokens", "lr"]
        columns += [f"weight_{key}" for key in config.sources]
        columns += [f"sequences_{key}" for key in config.sources]
        rows = []
        for step in range(config.steps):
            sequences = np.bincount(source_ids[step], minlength=len(config.sources))
            rows.append(
                [step, step * config.tokens_per_step, schedule.lr_at(config, step)]
                + list(datamodule.weights[step])
                + list(sequences)
            )
        csv_path = Path(self.logger.experiment.dir) / "schedule.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)  # missing when W&B is disabled
        with csv_path.open("w", newline="") as file:
            csv.writer(file).writerows([columns, *rows])
        artifact = wandb.Artifact(f"schedule-{config.name}", type="schedule")
        artifact.add_file(str(csv_path), policy="immutable")  # the run dir outlives the upload
        self.logger.experiment.log_artifact(artifact)

        stride = math.ceil(len(rows) / wandb.Table.MAX_ROWS)
        table = wandb.Table(columns=columns, data=rows[::stride])  # pyright: ignore[reportArgumentType]
        self.logger.experiment.log(
            {"schedule/table": table, "schedule/lr": wandb.plot.line(table, "step", "lr")}
        )


class SourceValLoss(L.Callback):
    """Validation loss per source, and their mean."""

    def __init__(self, source_keys: list[str]):
        self.source_keys = source_keys

    def on_validation_epoch_start(self, trainer, pl_module):
        self.loss_sums = torch.zeros(len(self.source_keys), device=pl_module.device)
        self.counts = torch.zeros(len(self.source_keys), device=pl_module.device)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_index, dataloader_idx=0
    ):
        self.loss_sums.index_add_(0, outputs["source_id"], outputs["per_sequence_loss"])
        self.counts.index_add_(
            0, outputs["source_id"], torch.ones_like(outputs["per_sequence_loss"])
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        losses = self.loss_sums / self.counts
        pl_module.log_dict({f"val/loss/{key}": loss for key, loss in zip(self.source_keys, losses)})
        pl_module.log("val/loss", losses.mean())


class Samples(L.Callback):
    """Text sampled from fixed prompts at each eval, as a W&B table."""

    def __init__(
        self, prompts: list[str], tokenizer: Tokenizer, new_tokens: int, logger: WandbLogger
    ):
        self.prompts = prompts
        self.logger = logger
        self.tokenizer = tokenizer
        self.new_tokens = new_tokens

    def on_validation_epoch_end(self, trainer, pl_module: LanguageModel):
        generator = torch.Generator(device=pl_module.device).manual_seed(0)
        rows = []
        for prompt in self.prompts:
            prompt_ids = [END_OF_TEXT_ID] + self.tokenizer.encode(prompt).ids
            prompt_tensor = torch.tensor([prompt_ids], device=pl_module.device)
            sample_ids = pl_module.model.generate(prompt_tensor, self.new_tokens, generator)
            rows.append(
                [trainer.global_step, prompt, self.tokenizer.decode(sample_ids[0].tolist())]
            )
        self.logger.log_text(key="samples", columns=["step", "prompt", "sample"], data=rows)
