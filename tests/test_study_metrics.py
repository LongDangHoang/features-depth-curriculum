import re

from tokenizers import Tokenizer

from features_depth_curriculum import data
from features_depth_curriculum.study.interface import load_facts, load_study
from features_depth_curriculum.study.metrics import subject_token_index
from features_depth_curriculum.study.run import run


def test_natural_answers_and_foils_are_single_tokens():
    tokenizer = Tokenizer.from_file(str(data.REPO_ROOT / "tokenizer.json"))
    _, probes = load_facts(["natural"])
    words = {word for probe in probes for word in [probe.target, *probe.foils]}
    assert {word for word in words if len(tokenizer.encode(word).ids) != 1} == set()


def test_example_study_gives_one_row_per_fact_per_metric():
    study = load_study("studies/example.yml")
    facts, _ = load_facts(["natural"])
    rows = run(study, "cpu", smoke=True)
    assert len(rows) == len(study.interventions) * len(study.metrics) * len(facts)


def test_subject_token_ends_the_subject():
    tokenizer = Tokenizer.from_file(str(data.REPO_ROOT / "tokenizer.json"))
    _, probes = load_facts(["natural"])
    for probe in probes:
        subject_match = re.search(rf" {re.escape(probe.subject)}\b", probe.prompt)
        assert subject_match, probe.id
        subject_end = subject_match.end()
        encoding = tokenizer.encode(probe.prompt)
        index = subject_token_index(tokenizer, probe.prompt, probe.subject)  # 1 = after end-of-text
        assert encoding.offsets[index - 1][1] == subject_end, probe.id
