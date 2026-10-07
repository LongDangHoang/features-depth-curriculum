from pathlib import Path

from features_depth_curriculum.config import load
from features_depth_curriculum.train import run


def test_cpu_smoke():
    cfg = load(Path(__file__).parents[1] / "experiments" / "example.yml").smoke()
    assert run(cfg, "cpu")["eval_loss"] >= 0
