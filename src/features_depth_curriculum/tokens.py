import hashlib
import itertools
import json
import random
import sys

import numpy as np
from tokenizers import Tokenizer

from features_depth_curriculum import data

TEMPLATE_BY_ROLE = {"base": "v0", "synthetic": "v0", "instruct": "v0"}
ENCODE_BATCH_DOCS = 10_000
WINDOW_TOKENS = 96
INTRO = """# PR 3: token files report

For each source and split, one flat file of `uint16` token ids (`.bin`, no header, no
compression), read with `np.memmap(path, dtype=np.uint16)`. Documents are concatenated in the
order of the clean jsonl file, each followed by `<|endoftext|>` (id 0). Next to it, a
`.idx.npy` holds the `int64` token offset at which each document starts.

File names are `<source>.<split>.<template>.bin`. The template version says how documents were
formatted before tokenising; v0 is the raw text for every role (the instruct source keeps its
original field labels). A new template version writes new files and leaves old ones alone.
`manifest.json` records the tokenizer sha1, and each file's counts and sha1, with the sha1 of the
clean file it came from. The tokenizer is the frozen `tokenizer.json` from PR 2.
"""


def format_document(role: str, template: str, doc: dict) -> str:
    """Return the text of a document as it is tokenised, for this role and template version."""
    if template == "v0":
        return doc["text"]
    raise NotImplementedError((role, template))


def tokenize_file(
    tokenizer: Tokenizer, source: data.Source, split: str, template: str, name: str
) -> dict:
    """Write one source and split as a .bin and .idx.npy; return its manifest entry."""
    end_of_text_id = tokenizer.token_to_id("<|endoftext|>")
    bin_dir = data.DATA_ROOT / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    doc_starts: list[int] = []
    total_tokens = 0
    with (bin_dir / f"{name}.bin").open("wb") as bin_file:
        for batch in itertools.batched(data.read_docs(source.key, split), ENCODE_BATCH_DOCS):
            texts = [format_document(source.role, template, doc) for doc in batch]
            batch_ids = []
            for encoding in tokenizer.encode_batch(texts):
                doc_starts.append(total_tokens + len(batch_ids))
                batch_ids += encoding.ids + [end_of_text_id]
            np.array(batch_ids, dtype=np.uint16).tofile(bin_file)
            total_tokens += len(batch_ids)
    np.save(bin_dir / f"{name}.idx.npy", np.array(doc_starts, dtype=np.int64))
    with (bin_dir / f"{name}.bin").open("rb") as bin_file:
        bin_sha1 = hashlib.file_digest(bin_file, "sha1").hexdigest()
    return {
        "role": source.role,
        "docs": len(doc_starts),
        "tokens": total_tokens,
        "sha1": bin_sha1,
    }


def render(tokenizer: Tokenizer, manifest: dict) -> str:
    """Render the report: counts table, one decoded window per file, upload command."""
    lines = [f"Tokenizer sha1: `{manifest['tokenizer_sha1']}`", ""]
    lines += ["| file | docs | tokens | tokens/doc | end-of-text | eot = docs |"]
    lines += ["|---|---:|---:|---:|---:|---|"]
    windows = []
    for name, entry in manifest["files"].items():
        tokens = np.memmap(data.DATA_ROOT / "bin" / f"{name}.bin", dtype=np.uint16)
        end_of_text = int(np.count_nonzero(tokens == 0))
        matches = "yes" if end_of_text == entry["docs"] else "no"
        tokens_per_doc = entry["tokens"] / entry["docs"]
        row = f"| {name} | {entry['docs']:,} | {entry['tokens']:,} | {tokens_per_doc:.1f} "
        lines += [row + f"| {end_of_text:,} | {matches} |"]
        start = random.Random(name).randrange(entry["tokens"])
        window_ids = tokens[start : start + WINDOW_TOKENS].tolist()
        window_text = tokenizer.decode(window_ids, skip_special_tokens=False)
        windows += [f"{name}, tokens {start:,} to {start + len(window_ids):,}:", ""]
        windows += ["~~~", window_text, "~~~", ""]
    lines += ["", "Random decoded windows (seeded by file name):", ""] + windows
    lines += [
        "## Handoff",
        "",
        "~~~",
        "modal volume put features-depth-curriculum data/bin /bin",
        "~~~",
    ]
    lines += ["", "Add `--force` to replace files already on the Volume.", ""]
    return INTRO + "\n" + "\n".join(lines)


def build() -> None:
    """Tokenise every source and split, then write the manifest and the report."""
    tokenizer_path = data.REPO_ROOT / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    clean_manifest = json.loads((data.DATA_ROOT / "manifests" / "clean.json").read_text("utf-8"))
    manifest = {
        "tokenizer_sha1": hashlib.sha1(tokenizer_path.read_bytes()).hexdigest(),
        "files": {},
    }
    for source in data.load_registry():
        template = TEMPLATE_BY_ROLE[source.role]
        for split in ("train", "val"):
            name = f"{source.key}.{split}.{template}"
            entry = tokenize_file(tokenizer, source, split, template, name)
            assert entry["docs"] == clean_manifest[source.key][split]["docs"]
            entry["clean_sha1"] = clean_manifest[source.key][split]["sha1"]
            manifest["files"][name] = entry
            print(f"{name}: docs={entry['docs']} tokens={entry['tokens']}")
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (data.DATA_ROOT / "bin" / "manifest.json").write_text(manifest_text, encoding="utf-8")
    (data.REPO_ROOT / "reports").mkdir(exist_ok=True)
    report = render(tokenizer, manifest)
    (data.REPO_ROOT / "reports" / "pr3_tokens.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    {"build": build}[sys.argv[1]]()
