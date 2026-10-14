import hashlib
import json
import random
import statistics
import sys

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from features_depth_curriculum import data

VOCAB_SIZE = 8192
SPECIAL_TOKENS = ["<|endoftext|>"] + [f"<|reserved_{index}|>" for index in range(1, 16)]
BUDGET_CHARS = 20_000_000
LOW_RATIO = 0.8
SHOWN_TOKENS = 60
INTRO = """# PR 2: tokenizer report

One byte-level BPE tokenizer with an 8192-token vocabulary, frozen for the whole study. It is
trained on train splits only, on an equal text budget per source: each source contributes a random
sample of about 20M characters (documents kept with probability budget / train chars, seed 0), or
its whole train split when that is smaller, so no single source dominates the vocabulary.
There is no normalizer (stage 1 already applied NFKC) and no post-processor (documents get
`<|endoftext|>` appended at packing time).

Ids 0..15 are fixed special tokens: `<|endoftext|>` at id 0, then `<|reserved_1|>` to
`<|reserved_15|>` for the future instruct template and role markers. Ids 16..271 are the 256 bytes.

Round trip means `decode(encode(text)) == text` on every val document. A source is flagged `low`
when its characters per token is below 0.8 times the median across sources.
"""


def sample_train_texts(source_key: str, train_chars: int) -> list[str]:
    """Keep each train document with probability budget / train chars, seed 0 per source."""
    rng = random.Random(0)
    keep_probability = min(1, BUDGET_CHARS / train_chars)
    return [
        doc["text"]
        for doc in data.read_docs(source_key, "train")
        if rng.random() < keep_probability
    ]


def train_tokenizer(samples: list[list[str]]) -> Tokenizer:
    """Train the byte-level BPE on all sources' samples, in registry order."""
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator((text for texts in samples for text in texts), trainer)
    return tokenizer


def evaluate_source(tokenizer: Tokenizer, source_key: str) -> dict:
    """Encode the val documents of one source; count tokens and round-trip failures."""
    stats = {"val_docs": 0, "val_chars": 0, "val_tokens": 0, "round_trip_failures": 0}
    for doc in data.read_docs(source_key, "val"):
        ids = tokenizer.encode(doc["text"]).ids
        stats["val_docs"] += 1
        stats["val_chars"] += len(doc["text"])
        stats["val_tokens"] += len(ids)
        stats["round_trip_failures"] += tokenizer.decode(ids) != doc["text"]
    return stats


def render(tokenizer: Tokenizer, manifest: dict, source_stats: dict) -> str:
    """Render the report: hash, special tokens, per-source table, sample of learned tokens."""
    chars_per_token = {
        key: stats["val_chars"] / stats["val_tokens"] for key, stats in source_stats.items()
    }
    median = statistics.median(chars_per_token.values())
    lines = [f"Tokenizer sha1: `{manifest['sha1']}`", ""]
    lines += [f"Vocabulary size: {tokenizer.get_vocab_size():,}", ""]
    special_ids = (f"`{token}`={tokenizer.token_to_id(token)}" for token in SPECIAL_TOKENS)
    lines += ["Special tokens: " + ", ".join(special_ids)]
    lines += [
        "",
        "| source | train chars sampled | val docs | val chars | val tokens | chars/token | round-trip failures | flag |",
    ]
    lines += ["|---|---:|---:|---:|---:|---:|---:|---|"]
    for key, stats in source_stats.items():
        flag = "low" if chars_per_token[key] < LOW_RATIO * median else ""
        trained = manifest["sources"][key]["training_chars"]
        row = (
            f"| {key} | {trained:,} | {stats['val_docs']:,} | {stats['val_chars']:,} "
            f"| {stats['val_tokens']:,} | {chars_per_token[key]:.3f} "
            f"| {stats['round_trip_failures']} | {flag} |"
        )
        lines += [row]
    learned_ids = range(len(SPECIAL_TOKENS) + 256, tokenizer.get_vocab_size())
    sampled_ids = sorted(random.Random(0).sample(learned_ids, SHOWN_TOKENS))
    shown = ", ".join(f"`{tokenizer.decode([token_id])!r}`" for token_id in sampled_ids)
    lines += ["", "Random learned tokens (seed 0, by id, so earlier merges first):", "", shown]
    return INTRO + "\n" + "\n".join(lines) + "\n"


def build() -> None:
    """Train and save the tokenizer, then write the manifest and the report."""
    clean_manifest = json.loads((data.DATA_ROOT / "manifests" / "clean.json").read_text("utf-8"))
    source_keys = [source.key for source in data.load_registry()]
    samples = [
        sample_train_texts(key, clean_manifest[key]["train"]["chars"]) for key in source_keys
    ]
    tokenizer = train_tokenizer(samples)
    tokenizer_path = data.REPO_ROOT / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    manifest = {
        "sha1": hashlib.sha1(tokenizer_path.read_bytes()).hexdigest(),
        "sources": {
            key: {
                "clean_train_sha1": clean_manifest[key]["train"]["sha1"],
                "training_chars": sum(map(len, texts)),
            }
            for key, texts in zip(source_keys, samples)
        },
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (data.DATA_ROOT / "manifests" / "tokenizer.json").write_text(manifest_text, encoding="utf-8")
    source_stats = {}
    for key in source_keys:
        stats = source_stats[key] = evaluate_source(tokenizer, key)
        print(
            f"{key}: train_chars={manifest['sources'][key]['training_chars']} "
            f"val_chars={stats['val_chars']} val_tokens={stats['val_tokens']} "
            f"round_trip_failures={stats['round_trip_failures']}"
        )
    (data.REPO_ROOT / "reports").mkdir(exist_ok=True)
    report = render(tokenizer, manifest, source_stats)
    (data.REPO_ROOT / "reports" / "pr2_tokenizer.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    {"build": build}[sys.argv[1]]()
