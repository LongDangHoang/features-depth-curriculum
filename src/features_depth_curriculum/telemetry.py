"""Perf telemetry: phase timestamps, per-eval-cycle durations and a profiler trace around an eval."""

import os
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger

from .config import Config

PHASES: dict[str, float] = {}


def process_start_epoch() -> float:
    """Wall-clock epoch at which this process started (Linux /proc)."""
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    stat = Path("/proc/self/stat").read_text()
    start_ticks = int(stat.rsplit(")", 1)[1].split()[19])  # field 22, after the "(comm)" field
    boot_lines = Path("/proc/stat").read_text().splitlines()
    boot_epoch = int(next(line for line in boot_lines if line.startswith("btime")).split()[1])
    return boot_epoch + start_ticks / ticks_per_second


PROCESS_START = process_start_epoch()


def mark(name: str) -> None:
    """Record the seconds since process start at which phase name first happened."""
    PHASES.setdefault(name, time.time() - PROCESS_START)


def finish(logger: WandbLogger) -> None:
    """Put the phases in the W&B summary, finish the run (final uploads), print the phase table."""
    for name, seconds in PHASES.items():
        logger.experiment.summary[f"phase/{name}"] = seconds
    logger.experiment.finish()
    mark("wandb_finish_done")
    # If the sandbox fakes /proc times, the offsets are wrong but the deltas still hold
    print(f"process start: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(PROCESS_START))}")
    previous = 0.0
    for name, seconds in PHASES.items():
        print(f"{name:28s} {seconds:9.1f}s  +{seconds - previous:8.1f}s")
        previous = seconds


def synchronize(pl_module: L.LightningModule) -> None:
    if pl_module.device.type == "cuda":
        torch.cuda.synchronize(pl_module.device)


class PhaseMarks(L.Callback):
    """Phase marks of the fit, and per eval cycle the checkpoint and validation durations.

    ModelCheckpoint saves in on_train_batch_end, then the loop runs validation (Lightning 2.6
    training_epoch_loop), so last batch end -> validation start is the synchronous save. Lightning
    sorts ModelCheckpoint after other callbacks, so this callback's batch end precedes the save.
    The first cycle also includes building the val dataloader.
    """

    def __init__(self, config: Config):
        self.eval_interval_steps = config.eval_interval_steps
        self.last_batch_end: float | None = None
        self.validation_start = 0.0

    def on_fit_start(self, trainer, pl_module):
        mark("fit_start")

    def on_train_start(self, trainer, pl_module):
        mark("train_start")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_index):
        if "first_step_done" not in PHASES or trainer.global_step % self.eval_interval_steps == 0:
            synchronize(pl_module)  # so the step's GPU tail is not billed to the checkpoint
        mark("first_step_done")
        self.last_batch_end = time.time()

    def on_validation_start(self, trainer, pl_module):
        self.validation_start = time.time()
        if self.last_batch_end is None:  # resumed from a save made before its validation
            return
        trainer.logger.log_metrics(  # pyright: ignore[reportOptionalMemberAccess]
            {"time/checkpoint_s": self.validation_start - self.last_batch_end}, trainer.global_step
        )

    def on_validation_end(self, trainer, pl_module):
        synchronize(pl_module)
        trainer.logger.log_metrics(  # pyright: ignore[reportOptionalMemberAccess]
            {"time/validation_s": time.time() - self.validation_start}, trainer.global_step
        )
        mark("first_validation_done")

    def on_train_end(self, trainer, pl_module):
        mark("train_end")


class GpuTrace(L.Callback):
    """torch.profiler trace around the second eval/checkpoint boundary (the first compiles eval mode)."""

    def __init__(self, config: Config):
        self.start_step = max(2 * config.eval_interval_steps - 5, 0)
        self.stop_step = min(2 * config.eval_interval_steps + 5, config.steps)
        self.profiler: torch.profiler.profile | None = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_index):
        if trainer.global_step != self.start_step:
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if pl_module.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(activities=activities)
        self.profiler.start()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_index):
        if trainer.global_step != self.stop_step or self.profiler is None:
            return
        synchronize(pl_module)
        self.profiler.stop()
        mark("trace_export_start")
        run_dir = Path(trainer.logger.experiment.dir)  # pyright: ignore[reportOptionalMemberAccess]
        run_dir.mkdir(parents=True, exist_ok=True)  # missing when W&B is disabled
        trace_path = run_dir / "trace.json.gz"
        self.profiler.export_chrome_trace(str(trace_path))
        sort_by = "cuda_time_total" if pl_module.device.type == "cuda" else "cpu_time_total"
        table = self.profiler.key_averages().table(sort_by=sort_by, row_limit=40)
        (run_dir / "key_averages.txt").write_text(table)
        print(table)
        self.profiler = None
        mark("trace_export_done")  # files in the run dir are uploaded at finish
