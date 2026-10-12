import hashlib
import json
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from features_depth_curriculum import data

LONG = "a document that is certainly long enough to survive the minimum length filter"
BABYLM_HEADERED = (
    f"= = = first title = = =\n{LONG} one\n= = = second title = = =\n{LONG} two\n"
    "= = = third title = = =\nshort\n"
)
BABYLM_PLAIN = f"first line {LONG}\nsecond line {LONG}\nshort\n"
TINYSTORIES = f"Words: a, b\nStory:\n{LONG}.\n\nSecond paragraph.\n<|endoftext|>\nStory: {LONG}\n<|endoftext|>\n"
REGISTRY = """sources:
  - {key: babylm_plain, role: base, hf_id: x/b, revision: r, format: babylm, files: ['plain.txt']}
  - {key: babylm_headered, role: base, hf_id: x/b, revision: r, format: babylm, files: ['headered.txt']}
  - {key: simplestories, role: synthetic, hf_id: x/s, revision: r, format: simplestories, files: ['*.parquet']}
  - {key: tinystories, role: instruct, hf_id: x/t, revision: r, format: tinystories_instruct, files: ['*.txt']}
"""


def test_is_validation_set_known_values():
    assert data.is_validation_set("simplestories", "story 26")
    assert data.is_validation_set("simplestories", "story 29")
    assert not data.is_validation_set("simplestories", "story 0")
    assert not data.is_validation_set("other_source", "x")


def test_normalize_removes_c0_and_c1_control_characters():
    assert data.normalize("a\x92b\x01c") == "abc"


def test_build_regenerates_identical_files(tmp_path, monkeypatch):
    for source_key in ("babylm_plain", "babylm_headered", "simplestories", "tinystories"):
        (tmp_path / "raw" / source_key).mkdir(parents=True)
    (tmp_path / "raw/babylm_plain/plain.txt").write_text(BABYLM_PLAIN, encoding="utf-8")
    (tmp_path / "raw/babylm_headered/headered.txt").write_text(BABYLM_HEADERED, encoding="utf-8")
    stories = {
        "story": [f"{LONG} {index}" for index in range(300)] + ["tiny"],
        "theme": ["t"] * 301,
    }
    pq.write_table(pa.table(stories), tmp_path / "raw/simplestories/stories.parquet")
    (tmp_path / "raw/tinystories/stories.txt").write_text(TINYSTORIES, encoding="utf-8")
    (tmp_path / "sources.yaml").write_text(REGISTRY, encoding="utf-8")
    monkeypatch.setattr(data, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(data, "REGISTRY", tmp_path / "sources.yaml")

    data.build()
    snapshot = {
        path: path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
        and "raw" not in path.relative_to(tmp_path).parts
        and path.name != "sources.yaml"
    }
    manifest = json.loads((tmp_path / "manifests/clean.json").read_text(encoding="utf-8"))
    for source_stats in manifest.values():
        accounted = source_stats["train"]["docs"] + source_stats["val"]["docs"]
        assert source_stats["parsed"] == accounted + sum(source_stats["dropped"].values())
    for source_key, source_stats in manifest.items():
        for split in ("train", "val"):
            output = (tmp_path / "clean" / source_key / f"{split}.jsonl").read_bytes()
            assert source_stats[split]["sha1"] == hashlib.sha1(output).hexdigest()

    for child in tmp_path.iterdir():
        if child.name not in ("raw", "sources.yaml"):
            shutil.rmtree(child)
    data.build()
    assert snapshot
    for path, content in snapshot.items():
        assert path.read_bytes() == content
