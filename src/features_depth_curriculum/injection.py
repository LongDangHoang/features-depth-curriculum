"""Injecting made-up facts into training (design.md, Scenarios: the injection schedules).

How it works. Each injection of the config becomes one extra source of the mixture, next to the
real corpus sources. Its tokens are the training text of its facts (study/facts/<file>_texts.yml),
packed in one stream like the corpus, and drawn like any other source: the trainer reads
seq_len-token chunks of it. Its weight at a step follows the schedule (see schedule.weights_at).
So there is no new data path: batches stay a pure function of (seed, step), and a run without
injections trains exactly as before.

One exposure of a fact is one reading of one of its texts. A fact has about 20 texts (40 if it is
a foundational fact with dependent stories), and one drawn sequence holds many texts, so
`sequence_draws` converts the wanted exposures into a number of drawn sequences for the whole run.
"""

from pathlib import Path

import numpy as np
import yaml
from jaxtyping import Int
from tokenizers import Tokenizer

from .config import Injection

FACTS_DIR = Path(__file__).parent / "study" / "facts"
MAX_INJECTED_SHARE = 0.5  # no step draws more than this share of its sequences from injections


def fact_texts(facts_file: str) -> dict[str, list[str]]:
    """The training text of each fact of a fact file: its texts, then its dependent stories."""
    entries = yaml.safe_load((FACTS_DIR / f"{facts_file}_texts.yml").read_text(encoding="utf-8"))
    return {
        fact_id: entry["texts"] + entry.get("dependents", []) for fact_id, entry in entries.items()
    }


def injected_fact_ids(injection: Injection) -> list[str]:
    """The ids of the injection's facts: the listed ones, or every fact of its fact type."""
    if injection.fact_ids:
        return injection.fact_ids
    spec = yaml.safe_load((FACTS_DIR / f"{injection.facts_file}.yml").read_text(encoding="utf-8"))
    return [fact["id"] for fact in spec["facts"] if fact["fact_type"] == injection.fact_type]


def pack_texts(texts: list[str], tokenizer: Tokenizer, seed: int) -> Int[np.ndarray, " tokens"]:
    """The texts in a random order, each followed by <|endoftext|>, as one stream of token ids."""
    end_of_text_id = tokenizer.token_to_id("<|endoftext|>")
    assert end_of_text_id is not None
    order = np.random.default_rng([seed, 2]).permutation(len(texts))
    token_ids: list[int] = []
    for encoding in tokenizer.encode_batch([texts[position] for position in order]):
        token_ids += encoding.ids + [end_of_text_id]
    return np.array(token_ids, dtype=np.uint16)


def sequence_draws(exposures: int, n_facts: int, n_texts: int, stream_tokens: int, seq_len: int):
    """How many sequences to draw from a packed stream so each fact is seen `exposures` times.

    One sequence reads seq_len tokens, so it reads seq_len * n_texts / stream_tokens texts on
    average. Every fact needs `exposures` text readings: n_facts * exposures in all.
    """
    return exposures * n_facts * stream_tokens / (seq_len * n_texts)


def build_injection(
    injection: Injection, tokenizer: Tokenizer, seq_len: int, seed: int
) -> tuple[Int[np.ndarray, " tokens"], float]:
    """The token stream of an injection, and the sequences to draw from it over the whole run."""
    texts_by_fact = fact_texts(injection.facts_file)
    fact_ids = injected_fact_ids(injection)
    texts = [text for fact_id in fact_ids for text in texts_by_fact[fact_id]]
    stream = pack_texts(texts, tokenizer, seed)
    draws = sequence_draws(injection.exposures, len(fact_ids), len(texts), len(stream), seq_len)
    # Few texts can be shorter than one sequence: repeat them. The texts per token stay the same.
    return np.tile(stream, -(-(seq_len + 1) // len(stream))), draws
