import hashlib
import json

from tokenizers import Tokenizer

from features_depth_curriculum import data, tokenizer

TRAIN_TEXT = "the quick brown fox jumps over the lazy dog, naïve café — ok\n\n  x"
VAL_TEXT = "naïve café — ok\n\n  x"
REGISTRY = """sources:
  - {key: first, role: base, hf_id: x/a, revision: r, format: babylm, files: ['a.txt']}
  - {key: second, role: base, hf_id: x/b, revision: r, format: babylm, files: ['b.txt']}
"""


def test_build_fixes_special_ids_round_trips_and_is_deterministic(tmp_path, monkeypatch):
    clean_manifest = {}
    for source_key in ("first", "second"):
        source_dir = tmp_path / "clean" / source_key
        source_dir.mkdir(parents=True)
        train_lines = [
            json.dumps({"text": f"{TRAIN_TEXT} {index}", "meta": {}}) for index in range(50)
        ]
        (source_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
        val_line = json.dumps({"text": VAL_TEXT, "meta": {}}, ensure_ascii=False)
        (source_dir / "val.jsonl").write_text(val_line + "\n", encoding="utf-8")
        train_chars = sum(len(f"{TRAIN_TEXT} {index}") for index in range(50))
        clean_manifest[source_key] = {"train": {"chars": train_chars, "sha1": "fake"}}
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "clean.json").write_text(json.dumps(clean_manifest), encoding="utf-8")
    (tmp_path / "sources.yaml").write_text(REGISTRY, encoding="utf-8")
    monkeypatch.setattr(data, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(data, "REGISTRY", tmp_path / "sources.yaml")

    tokenizer.build()
    tokenizer_path = tmp_path / "tokenizer.json"
    first_bytes = tokenizer_path.read_bytes()
    manifest = json.loads((tmp_path / "manifests" / "tokenizer.json").read_text(encoding="utf-8"))
    assert manifest["sha1"] == hashlib.sha1(first_bytes).hexdigest()
    loaded = Tokenizer.from_file(str(tokenizer_path))
    assert [loaded.token_to_id(token) for token in tokenizer.SPECIAL_TOKENS] == list(range(16))
    assert loaded.decode(loaded.encode(VAL_TEXT).ids) == VAL_TEXT

    tokenizer.build()
    assert tokenizer_path.read_bytes() == first_bytes
