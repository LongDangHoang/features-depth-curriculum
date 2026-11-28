import re

import yaml
from tokenizers import Tokenizer

from features_depth_curriculum import data
from features_depth_curriculum.study.interface import FACTS_DIR, load_facts, load_study
from features_depth_curriculum.study.run import run


def test_natural_answers_and_foils_are_single_tokens():
    tokenizer = Tokenizer.from_file(str(data.REPO_ROOT / "tokenizer.json"))
    _, probes = load_facts(["natural"])
    words = {
        word
        for probe in probes
        if probe.kind != "statement"
        for word in [probe.target, *probe.foils]
    }
    assert {word for word in words if len(tokenizer.encode(word).ids) != 1} == set()


def test_example_study_gives_one_row_per_fact_per_metric():
    study = load_study("studies/example.yml")
    facts, _ = load_facts(["natural"])
    rows = run(study, "cpu", smoke=True)
    assert len(rows) == len(study.interventions) * len(study.metrics) * len(facts)


def test_statement_forms_and_templates_are_clean():
    spec = yaml.safe_load((FACTS_DIR / "natural.yml").read_text(encoding="utf-8"))
    subjects = {entry["subject"] for entry in spec["facts"]}
    for relation, templates in spec["statements"].items():
        for template_id, template in templates.items():
            assert len(template["text"].split("{form}")[-1].split()) >= 3, template_id
            template_words = set(re.findall(r"[a-z]+", template["text"].lower()))
            assert not template_words & subjects, template_id
        all_forms = []
        for answer, forms_of_answer in spec["forms"][relation].items():
            words_of_split = {
                split: {word for form in forms for word in form.split()}
                for split, forms in forms_of_answer.items()
            }
            assert not words_of_split["dev"] & words_of_split["test"], answer
            assert not (words_of_split["dev"] | words_of_split["test"]) & subjects, answer
            all_forms += forms_of_answer["dev"] + forms_of_answer["test"]
        assert len(all_forms) == len(set(all_forms)), relation
