import hashlib
import json

import numpy as np
from tokenizers import Tokenizer

from features_depth_curriculum import data, tokenizer, tokens

DOCS = {
    "first": ["the quick brown fox jumps over the lazy dog", "naïve café — ok", "a dog"],
    "second": ["Words: fox, dog\nStory: the fox met the dog", "Words: cat\nStory: a cat sat"],
}
REGISTRY = """sources:
  - {key: first, role: base, hf_id: x/a, revision: r, format: babylm, files: ['a.txt']}
  - {key: second, role: instruct, hf_id: x/b, revision: r, format: babylm, files: ['b.txt']}
"""


def test_build_writes_end_of_text_terminated_files_with_offsets(tmp_path, monkeypatch):
    clean_manifest = {}
    for source_key, texts in DOCS.items():
        source_dir = tmp_path / "clean" / source_key
        source_dir.mkdir(parents=True)
        lines = [json.dumps({"text": text, "meta": {}}, ensure_ascii=False) for text in texts]
        for split in ("train", "val"):
            (source_dir / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        train_chars = sum(map(len, texts))
        clean_manifest[source_key] = {
            "train": {"chars": train_chars, "docs": len(texts), "sha1": "fake"},
            "val": {"chars": train_chars, "docs": len(texts), "sha1": "fake"},
        }
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "clean.json").write_text(json.dumps(clean_manifest), encoding="utf-8")
    (tmp_path / "sources.yaml").write_text(REGISTRY, encoding="utf-8")
    monkeypatch.setattr(data, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(data, "REGISTRY", tmp_path / "sources.yaml")

    tokenizer.build()
    tokens.build()
    loaded = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    bin_dir = tmp_path / "bin"
    token_ids = np.fromfile(bin_dir / "second.val.v0.bin", dtype=np.uint16)
    expected = "".join(text + "<|endoftext|>" for text in DOCS["second"])
    assert loaded.decode(token_ids.tolist(), skip_special_tokens=False) == expected
    doc_starts = np.load(bin_dir / "second.val.v0.idx.npy")
    assert doc_starts[0] == 0 and len(doc_starts) == len(DOCS["second"])
    assert all(token_ids[start - 1] == 0 for start in doc_starts[1:])
    manifest = json.loads((bin_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["tokenizer_sha1"]
        == hashlib.sha1((tmp_path / "tokenizer.json").read_bytes()).hexdigest()
    )
