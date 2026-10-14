import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FDC_DATA", REPO_ROOT / "data"))
REGISTRY = REPO_ROOT / "sources.yaml"
CHUNK_CHARS = 4000
MIN_CHARS = 50
TSI_FIELDS = re.compile(r"^(Words|Features|Summary|Random sentence|Story): ?", re.MULTILINE)
BABYLM_HEADER = re.compile(r"^= = = .* = = =\n?", re.MULTILINE)
INTRO = """# PR 1: data report

Per document: Unicode NFKC, CRLF to LF, C0 and C1 control characters removed, trailing spaces
stripped, runs of spaces collapsed, 3+ newlines collapsed to 2, then stripped. Documents shorter
than 50 chars are dropped, then exact duplicates (within a source, first one kept).
Split: val if `int(sha1(key + text)[:8], 16) % 1000 < 10`.

BabyLM is the 2026 Strict 100M track (cased, detoxified upstream), one source per domain file.
Files are split into documents on their `= = = title = = =` header lines (title dropped), then
into 4000-char line-boundary chunks; the split is per chunk, so val is chunk-disjoint, not
document-disjoint. Headers mark CHILDES transcripts, Gutenberg books and Simple Wikipedia
articles; BNC, OpenSubtitles and Switchboard have none, so they are only chunked. Speaker tags
(`*MOT:`, `A:`) and CHILDES `[action]` lines are kept as upstream. Upstream files are not
perfectly separated by domain, and this is kept as is: open_subtitles contains about 3,800 short
Simple Wikipedia articles, childes contains 8 Gutenberg books, simple_wiki contains one 5 MB
Switchboard-style dialogue, and bnc_spoken ends with one CHILDES transcript.

SimpleStories and TinyStoriesInstruct ship their own train/test/valid files; these are merged,
then re-split by the hash rule. The sha1 of each output file is listed so a rerun can be checked.
"""


@dataclass
class Source:
    key: str
    role: str
    hf_id: str
    revision: str
    format: str
    files: list[str]


def load_registry() -> list[Source]:
    """Read sources.yaml into Source entries."""
    entries = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["sources"]
    return [Source(**entry) for entry in entries]


def is_validation_set(source_key: str, text: str) -> bool:
    """Return True for the 1% of documents the hash rule sends to val."""
    digest = hashlib.sha1((source_key + text).encode()).hexdigest()
    return int(digest[:8], 16) % 1000 < 10


def normalize(text: str) -> str:
    """Apply the Unicode and whitespace cleanup described in the report intro."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def raw_files(source: Source) -> list[Path]:
    """List the downloaded files matching the source's patterns."""
    raw_dir = DATA_ROOT / "raw" / source.key
    paths = [path for pattern in source.files for path in sorted(raw_dir.glob(pattern))]
    assert paths, f"no files for {source.key}"
    return paths


def chunk_lines(text: str, max_chars: int) -> Iterator[str]:
    """Split text into chunks of whole lines of at most max_chars (one longer line stays whole)."""
    chunk_parts: list[str] = []
    chunk_size = 0
    for line in text.splitlines(keepends=True):
        if chunk_parts and chunk_size + len(line) > max_chars:
            yield "".join(chunk_parts)
            chunk_parts, chunk_size = [], 0
        chunk_parts.append(line)
        chunk_size += len(line)
    if chunk_parts:
        yield "".join(chunk_parts)


def parse_babylm(source: Source) -> Iterator[tuple[str, dict]]:
    """Yield chunks of each document of each BabyLM file, documents split on header lines."""
    for path in raw_files(source):
        for document in BABYLM_HEADER.split(path.read_text(encoding="utf-8")):
            for chunk in chunk_lines(document, CHUNK_CHARS):
                yield chunk, {}


def parse_simplestories(source: Source) -> Iterator[tuple[str, dict]]:
    """Yield each SimpleStories story with its other columns as metadata."""
    for path in raw_files(source):
        for batch in pq.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                yield row.pop("story"), row


def tsi_records(path: Path) -> Iterator[str]:
    """Yield the raw text of each <|endoftext|>-separated TinyStoriesInstruct record."""
    record_lines: list[str] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip() == "<|endoftext|>":
                yield "".join(record_lines)
                record_lines = []
            else:
                record_lines.append(line)
    yield "".join(record_lines)


def parse_tinystories_instruct(source: Source) -> Iterator[tuple[str, dict]]:
    """Yield each TinyStoriesInstruct record with its labelled fields as metadata."""
    for path in raw_files(source):
        for record in filter(str.strip, tsi_records(path)):
            _, *labels_and_values = TSI_FIELDS.split(record)
            meta = {
                label.lower().replace(" ", "_"): value.strip()
                for label, value in zip(labels_and_values[::2], labels_and_values[1::2])
            }
            yield record.strip(), meta


PARSERS = {
    "babylm": parse_babylm,
    "simplestories": parse_simplestories,
    "tinystories_instruct": parse_tinystories_instruct,
}


def build_source(source: Source) -> dict:
    """Parse, clean, dedup and split one source into train.jsonl and val.jsonl; return its stats."""
    counts: Counter = Counter()
    split_stats: dict[str, dict] = {split: {"docs": 0, "chars": 0} for split in ("train", "val")}
    seen_digests: set[bytes] = set()
    output_dir = DATA_ROOT / "clean" / source.key
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        (output_dir / "train.jsonl").open("w", encoding="utf-8") as train_file,
        (output_dir / "val.jsonl").open("w", encoding="utf-8") as val_file,
    ):
        split_files = {"train": train_file, "val": val_file}
        for text, meta in PARSERS[source.format](source):
            counts["parsed"] += 1
            text = normalize(text)
            if len(text) < MIN_CHARS:
                counts["too_short"] += 1
                continue
            digest = hashlib.sha1(text.encode()).digest()
            if digest in seen_digests:
                counts["duplicate"] += 1
                continue
            seen_digests.add(digest)
            split = "val" if is_validation_set(source.key, text) else "train"
            line = json.dumps({"text": text, "meta": meta}, ensure_ascii=False)
            split_files[split].write(line + "\n")
            split_stats[split]["docs"] += 1
            split_stats[split]["chars"] += len(text)
    for split, stats in split_stats.items():
        with (output_dir / f"{split}.jsonl").open("rb") as file:
            stats["sha1"] = hashlib.file_digest(file, "sha1").hexdigest()
    dropped = {"too_short": counts["too_short"], "duplicate": counts["duplicate"]}
    return {
        "role": source.role,
        "hf_id": source.hf_id,
        "revision": source.revision,
        "parsed": counts["parsed"],
        "dropped": dropped,
        **split_stats,
    }


def read_docs(source_key: str, split: str) -> Iterator[dict]:
    """Yield the cleaned documents of one source and split."""
    with (DATA_ROOT / "clean" / source_key / f"{split}.jsonl").open(encoding="utf-8") as file:
        yield from map(json.loads, file)


def render_source(source_key: str, source_stats: dict) -> str:
    """Render one source's section of the report."""
    nonascii_counts: Counter = Counter()
    for split in ("train", "val"):
        for doc in read_docs(source_key, split):
            nonascii_counts.update(char for char in doc["text"] if not char.isascii())
    nonascii_total = sum(nonascii_counts.values())
    total_chars = source_stats["train"]["chars"] + source_stats["val"]["chars"]
    train_docs = source_stats["train"]["docs"]
    sampled_indices = set(random.Random(0).sample(range(train_docs), min(10, train_docs)))
    role, hf_id, revision = (source_stats[field] for field in ("role", "hf_id", "revision"))
    lines = [f"## {source_key} ({role}, `{hf_id}@{revision}`)", ""]
    lines += ["| split | docs | chars | sha1 |", "|---|---:|---:|---|"]
    lines += [
        f"| {split} | {source_stats[split]['docs']:,} | {source_stats[split]['chars']:,} "
        f"| `{source_stats[split]['sha1']}` |"
        for split in ("train", "val")
    ]
    lines += ["", f"Parsed {source_stats['parsed']:,} documents. Dropped: "]
    lines += [f"- {reason}: {count:,}" for reason, count in source_stats["dropped"].items()]
    ratio = nonascii_total / max(total_chars, 1)
    lines += ["", f"Non-ASCII: {nonascii_total:,} chars ({ratio:.4%}). Top:", ""]
    lines += [
        f"- `{char}` U+{ord(char):04X} {unicodedata.name(char, '?')}: {count:,}"
        for char, count in nonascii_counts.most_common(10)
    ]
    lines += ["", "Random train documents (seed 0):", ""]
    for index, doc in enumerate(read_docs(source_key, "train")):
        if index in sampled_indices:
            text = doc["text"][:400] + (" [...]" if len(doc["text"]) > 400 else "")
            lines += ["~~~", text, "~~~"]
            if doc["meta"]:
                short_meta = {key: str(value)[:80] for key, value in doc["meta"].items()}
                lines += [f"meta: `{json.dumps(short_meta, ensure_ascii=False)}`"]
            lines += [""]
    return "\n".join(lines)


def download() -> None:
    """Download every registered source at its pinned revision."""
    for source in load_registry():
        snapshot_download(
            source.hf_id,
            repo_type="dataset",
            revision=source.revision,
            allow_patterns=source.files,
            local_dir=DATA_ROOT / "raw" / source.key,
        )


def build() -> None:
    """Build every source, then write the manifest and the report."""
    manifest = {}
    for source in load_registry():
        stats = manifest[source.key] = build_source(source)
        print(
            f"{source.key}: parsed={stats['parsed']} too_short={stats['dropped']['too_short']} "
            f"duplicate={stats['dropped']['duplicate']} "
            f"train={stats['train']['docs']} val={stats['val']['docs']}"
        )
    (DATA_ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (DATA_ROOT / "manifests" / "clean.json").write_text(manifest_text, encoding="utf-8")
    (REPO_ROOT / "reports").mkdir(exist_ok=True)
    sections = [render_source(key, manifest[key]) for key in sorted(manifest)]
    report = INTRO + "\n" + "\n".join(sections)
    (REPO_ROOT / "reports" / "pr1_data.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    {"download": download, "build": build}[sys.argv[1]]()
