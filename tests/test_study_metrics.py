from tokenizers import Tokenizer

from features_depth_curriculum import data
from features_depth_curriculum.study.interface import load_facts, load_study
from features_depth_curriculum.study.run import run


def test_natural_answers_and_foils_are_single_tokens():
    tokenizer = Tokenizer.from_file(str(data.REPO_ROOT / "tokenizer.json"))
    _, probes = load_facts(["natural"])
    words = {word for probe in probes for word in [probe.target, *probe.foils]}
    assert {word for word in words if len(tokenizer.encode(word).ids) != 1} == set()


def test_example_study_gives_one_row_per_fact_per_metric():
    study = load_study("studies/example.yml")
    rows = run(study, "cpu", smoke=True)
    assert len(rows) == len(study.interventions) * len(study.metrics) * 14  # 14 natural facts
