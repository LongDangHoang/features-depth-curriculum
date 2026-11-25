"""Injecting made-up facts into training (design.md, Scenarios: the injection schedules).

How it works. Each injection of the config becomes extra sequences in the batches. Their number
is fixed by the exposures, and each is put at a fixed step and slot that follows the schedule
(`study.schedules.sequence_steps`, used by mixture.Mixture). The dose is exact, not random, and
a run without injections trains exactly as before.

One exposure of a fact is one reading of one of its texts. A fact has about 20 texts (about 40
with the dependent stories of a foundational fact). The injected sequences are windows of
seq_len + 1 tokens cut one after the other from a long stream. The stream is the injection's
texts in a shuffled order, again and again (a new shuffle each pass, seeded by the run seed and
the injection). So every text is read about equally often, also the last ones, and a window
starts wherever the stream is, as in the corpus.
"""

import numpy as np
import yaml
from jaxtyping import Int
from tokenizers import Tokenizer

from .config import FACTS_DIR, Config, Injection

MAX_INJECTED_SHARE = 0.5  # of a step's sequences, for the dose ladder (one step can hold more)


def fact_texts(facts_file: str) -> dict[str, list[str]]:
    """The training text of each fact of a fact file: its texts, then its dependent stories."""
    return {
        fact_id: entry["texts"] + entry.get("dependents", [])
        for fact_id, entry in load_texts(facts_file).items()
    }


def load_texts(facts_file: str) -> dict[str, dict[str, list[str]]]:
    return yaml.safe_load((FACTS_DIR / f"{facts_file}_texts.yml").read_text(encoding="utf-8"))


def pack_texts(texts: list[str], tokenizer: Tokenizer, seed: int) -> Int[np.ndarray, " tokens"]:
    """The texts in a random order, each followed by <|endoftext|>, as one stream of token ids."""
    end_of_text_id = tokenizer.token_to_id("<|endoftext|>")
    assert end_of_text_id is not None
    order = np.random.default_rng([seed, 2]).permutation(len(texts))
    token_ids: list[int] = []
    for encoding in tokenizer.encode_batch([texts[position] for position in order]):
        token_ids += encoding.ids + [end_of_text_id]
    return np.array(token_ids, dtype=np.uint16)


def build_windows(
    texts: list[str],
    exposures: int,
    n_facts: int,
    tokenizer: Tokenizer,
    seq_len: int,
    seed_key: list[int],
) -> Int[np.ndarray, "windows seq_plus_one"]:
    """The sequences that give each of n_facts facts `exposures` readings of its texts.

    n_facts * exposures readings of texts are needed, so the stream has that many texts in all,
    and it is cut into windows of seq_len + 1 tokens (one shared token between windows)."""
    end_of_text_id = tokenizer.token_to_id("<|endoftext|>")
    assert end_of_text_id is not None
    encoded = [encoding.ids + [end_of_text_id] for encoding in tokenizer.encode_batch(texts)]
    passes = exposures * n_facts / len(texts)
    n_windows = max(1, round(passes * sum(map(len, encoded)) / seq_len))
    rng = np.random.default_rng(seed_key)
    stream: list[int] = []
    while len(stream) < n_windows * seq_len + 1:
        for position in rng.permutation(len(encoded)):
            stream += encoded[position]
    stream_array = np.array(stream, dtype=np.uint16)
    return np.stack(
        [
            stream_array[start : start + seq_len + 1]
            for start in range(0, n_windows * seq_len, seq_len)
        ]
    )  # (n_windows, seq_len + 1)


def planned_exposures(config: Config) -> dict[str, dict[str, float]]:
    """From the config alone: the exposures planned for each injected fact, counted two ways.

    "texts": readings of its stating texts only. "texts_and_dependents": also its dependent
    stories (foundational facts). The study counts exposures as the second."""
    planned: dict[str, dict[str, float]] = {}
    for injection in config.injections:
        entries = load_texts(injection.facts_file)
        fact_ids = injection.selected_fact_ids()
        passes = (
            injection.exposures
            * len(fact_ids)
            / sum(
                len(entries[fact_id]["texts"]) + len(entries[fact_id].get("dependents", []))
                for fact_id in fact_ids
            )
        )
        for fact_id in fact_ids:
            counts = planned.setdefault(fact_id, {"texts": 0.0, "texts_and_dependents": 0.0})
            counts["texts"] += passes * len(entries[fact_id]["texts"])
            counts["texts_and_dependents"] += passes * (
                len(entries[fact_id]["texts"]) + len(entries[fact_id].get("dependents", []))
            )
    return planned


def build_injection(
    injection: Injection, tokenizer: Tokenizer, seq_len: int, seed: int, index: int
) -> Int[np.ndarray, "windows seq_plus_one"]:
    """The windows of the index-th injection of a config."""
    texts_by_fact = fact_texts(injection.facts_file)
    fact_ids = injection.selected_fact_ids()
    texts = [text for fact_id in fact_ids for text in texts_by_fact[fact_id]]
    return build_windows(
        texts, injection.exposures, len(fact_ids), tokenizer, seq_len, [seed, 2, index]
    )
