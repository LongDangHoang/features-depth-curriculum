"""Thin Modal wrapper. All logic lives in train.run() and the data build modules."""

import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # lets `modal run <path>` import the package without install

from features_depth_curriculum.config import GPUS, Config, load

image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_sync(str(ROOT), extra_options="--no-install-project")  # same uv.lock as the box
    .env({"PYTHONPATH": "/repo/src"})
    # Same layout as the repo, so data.REPO_ROOT is /repo and finds the registry and tokenizer
    .add_local_file(ROOT / "sources.yaml", "/repo/sources.yaml")
    .add_local_file(ROOT / "tokenizer.json", "/repo/tokenizer.json")
    .add_local_dir(ROOT / "src", "/repo/src", ignore=["**/__pycache__"])
)
app = modal.App("features-depth-curriculum", image=image)
# WANDB_API_KEY lives in a Modal-side secret, never in GitHub. Create it once, in the
# dedicated Modal workspace: modal secret create wandb WANDB_API_KEY=<key>
secrets = [modal.Secret.from_name("wandb")]
volume = modal.Volume.from_name("features-depth-curriculum")


# Real gpu/timeout are set per config via with_options.
@app.function(gpu=GPUS[0], timeout=300, secrets=secrets)
def train(cfg: dict) -> dict:
    from features_depth_curriculum.train import run

    return run(Config(**cfg), "cuda")


# CPU only: download, clean and tokenise every source straight onto the Volume.
# modal run --detach src/features_depth_curriculum/modal_app.py::build_data
@app.function(cpu=2, timeout=2 * 3600, volumes={"/vol": volume}, env={"FDC_DATA": "/vol"})
def build_data():
    from features_depth_curriculum import data, tokens

    data.download()
    data.build()
    tokens.build()
    volume.commit()
    print((data.DATA_ROOT / "bin" / "manifest.json").read_text())


@app.local_entrypoint()
def main(config: str, smoke: bool = False):
    cfg = load(config)
    if smoke:
        cfg = cfg.smoke(keep_gpu_features=True).model_copy(update={"gpu": GPUS[0]})
        print(train.with_options(gpu=cfg.gpu, timeout=300).remote(cfg.model_dump()))
        return
    fn = train.with_options(gpu=cfg.gpu, timeout=cfg.timeout_s, max_containers=cfg.max_containers)
    # spawn, not remote: with `modal run --detach` the call keeps running after we exit
    print("spawned", fn.spawn(cfg.model_dump()).object_id)
