import os

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from tokenizers import Tokenizer

from . import callbacks, data
from .config import Config
from .mixture import MixtureData
from .module import LanguageModel

PROMPTS = [
    "Once upon a time",
    "The little dog",
    "One day, a girl named",
    "She looked at the sky and",
]


def flops_per_token(config: Config, parameter_counts: dict[str, int]) -> int:
    """Training FLOPs per token: 6 per matmul parameter (non-embedding and head), plus attention."""
    head = config.width * config.vocab_size
    attention = 12 * config.depth * config.width * config.seq_len
    return 6 * (parameter_counts["non_embedding"] + head) + attention


def checkpoint_dir(name: str) -> str:
    """Where the checkpoints of the config called name live: on S3 if configured, else locally."""
    # The owner may move the layout to {project}/{name}/checkpoints
    if "S3_BUCKET" in os.environ:
        return f"s3://{os.environ['S3_BUCKET']}/checkpoints/{name}"
    return str(data.DATA_ROOT / "checkpoints" / name)


def run(config: Config, device: str) -> dict[str, float]:
    """Train (or resume, if a checkpoint of this name exists); return the final val losses."""
    L.seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")

    module = LanguageModel(config)
    parameter_counts = module.model.parameter_counts()
    print(f"parameters: {parameter_counts}")
    logger = WandbLogger(
        project="features-depth-curriculum",
        name=config.name,
        id=config.name,
        resume="allow",
        config=config.model_dump(),
    )
    logger.log_hyperparams(parameter_counts)

    datamodule = MixtureData(config)
    tokenizer = Tokenizer.from_file(str(data.REPO_ROOT / "tokenizer.json"))
    trainer = L.Trainer(
        accelerator=device,
        devices=1,
        precision="bf16-mixed" if config.bf16 else "32-true",
        max_steps=config.steps,
        val_check_interval=config.eval_interval_steps,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        gradient_clip_val=config.grad_clip,
        log_every_n_steps=callbacks.LOG_EVERY_N_STEPS,
        logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            callbacks.Throughput(
                flops_per_token(config, parameter_counts), callbacks.PEAK_BF16_FLOPS[config.gpu]
            ),
            callbacks.LossAndGradNorm(),
            callbacks.PlannedSchedule(config, datamodule, logger),
            callbacks.SourceValLoss(config.sources),
            callbacks.Samples(PROMPTS, tokenizer, config.sample_tokens, logger),
            ModelCheckpoint(
                dirpath=checkpoint_dir(config.name),
                every_n_train_steps=config.eval_interval_steps,
                save_top_k=0,
                save_last=True,
            ),
        ],
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path="last")
    return {
        key: float(value)
        for key, value in trainer.callback_metrics.items()
        if key.startswith("val/")
    }
