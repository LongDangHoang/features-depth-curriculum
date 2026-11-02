import torch
from tokenizers import Tokenizer

from features_depth_curriculum import data
from features_depth_curriculum.config import load
from features_depth_curriculum.model import GPT
from features_depth_curriculum.pilot import probes, removal, scoring
from features_depth_curriculum.pilot.messages import EvalLog, ModelRef, ProbeScore, summarise
from features_depth_curriculum.pilot.run import run


def test_probe_answers_are_single_tokens():
    tokenizer = Tokenizer.from_file(str(data.REPO_ROOT / "tokenizer.json"))
    _, all_probes = probes.load()
    words = {word for probe in all_probes for word in [probe.target, *probe.foils]}
    assert {word for word in words if len(tokenizer.encode(word).ids) != 1} == set()


def test_summary_round_trips_and_counts_wins():
    _, all_probes = probes.load()
    test_probes = [probe for probe in all_probes if probe.fact_id == "banana_yellow"]
    test_probes = [probe for probe in test_probes if probe.split == "test"]
    scores = [
        ProbeScore(
            probe_id=probe.id,
            target_logprob=-1.0 if index % 2 else -5.0,  # wins on every other probe
            foil_logprobs={foil: -3.0 for foil in probe.foils},
        )
        for index, probe in enumerate(test_probes)
    ]
    log = EvalLog(
        model=ModelRef(config_name="sanity", checkpoint="last"), interventions=[], scores=scores
    )
    log = EvalLog.model_validate_json(log.model_dump_json())
    [summary] = summarise(log, all_probes)
    assert summary.fact_id == "banana_yellow"
    assert summary.consistency == 0.5


def test_erase_removes_direction_from_residual_stream():
    config = load(data.REPO_ROOT / "experiments/sanity.yml").smoke()
    model = GPT(config)
    token_ids = torch.randint(0, config.vocab_size, (4, 10))
    last_index = torch.tensor([9, 5, 7, 9])
    direction = removal.random_direction(config.width, seed=0, device="cpu")
    removal.erase(model, direction)
    assert (scoring.residuals(model, token_ids, last_index) @ direction).abs().max() < 1e-4


def test_run_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(data, "DATA_ROOT", tmp_path)
    monkeypatch.setattr("features_depth_curriculum.pilot.run.STEPS", (1, 2, 3))  # fast on CPU
    config = load(data.REPO_ROOT / "experiments/sanity.yml").smoke()
    logs = run(config, "last", "cpu", smoke=True)
    assert len(logs) == 1 + 14 * config.depth + 3 + 3 * 14 * 3
    assert (tmp_path / "pilots" / "fact_removal" / f"{config.name}-last.jsonl").exists()
