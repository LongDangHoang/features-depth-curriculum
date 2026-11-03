import lightning as L
import torch

from features_depth_curriculum.config import Config
from features_depth_curriculum.module import LanguageModel, MuonWithAdamW, build_optimizer


def tiny_config() -> Config:
    return Config(
        name="tiny",
        sources=["s"],
        depth=2,
        width=32,
        heads=2,
        seq_len=16,
        vocab_size=64,
        batch_size=4,
        token_budget=64 * 40,
        warmup_tokens=64,
        decay_tokens=640,
        eval_interval_tokens=64,
        speedrun=True,
    )


def test_muon_gets_block_matrices_and_adamw_the_rest():
    module = LanguageModel(tiny_config())
    optimizer = build_optimizer(module.model, module.config)
    assert isinstance(optimizer, MuonWithAdamW)
    muon, adamw = optimizer.optimizers
    muon_params = [p for group in muon.param_groups for p in group["params"]]
    adamw_params = [p for group in adamw.param_groups for p in group["params"]]
    assert {id(p) for p in muon_params} | {id(p) for p in adamw_params} == {
        id(p) for p in module.model.parameters()
    }
    assert not {id(p) for p in muon_params} & {id(p) for p in adamw_params}
    assert len(muon_params) == 4 * module.config.depth  # qkv, proj, up, down per block


def test_speedrun_loss_goes_down_and_checkpoint_restores_both_optimizers():
    L.seed_everything(0)
    config = tiny_config()
    module = LanguageModel(config)
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len + 1))
    batch = {"tokens": tokens[:, :-1], "targets": tokens[:, 1:]}  # one fixed batch to overfit
    loader = torch.utils.data.DataLoader([batch] * config.steps, batch_size=None)  # pyright: ignore[reportArgumentType]
    trainer = L.Trainer(
        accelerator="cpu", max_steps=config.steps, gradient_clip_val=1.0, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    )  # fmt: skip
    loss_before = module.per_sequence_loss(batch["tokens"], batch["targets"]).mean().item()
    trainer.fit(module, loader)
    loss_after = module.per_sequence_loss(batch["tokens"], batch["targets"]).mean().item()
    assert loss_after < loss_before - 0.5

    optimizer = build_optimizer(module.model, config)
    assert isinstance(optimizer, MuonWithAdamW)
    optimizer.load_state_dict(trainer.optimizers[0].state_dict())
    assert optimizer.optimizers[0].state  # Muon momentum came back
    for group in optimizer.param_groups:  # as the lr scheduler does after a resume
        group["lr"] = 0.123
    inner_groups = [group for inner in optimizer.optimizers for group in inner.param_groups]
    assert all(group["lr"] == 0.123 for group in inner_groups)
