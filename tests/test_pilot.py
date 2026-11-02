from tokenizers import Tokenizer

from features_depth_curriculum import data
from features_depth_curriculum.pilot import probes
from features_depth_curriculum.pilot.messages import EvalLog, ModelRef, ProbeScore, summarise


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
