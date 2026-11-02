"""Thin Modal wrapper. All logic lives in train.run() and the data build modules."""

import sys
from pathlib import Path

import modal

# In the container Modal mounts this file alone at /root/modal_app.py, so parents[2] does not exist there
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/repo")
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
# Credentials live in Modal-side secrets, never in GitHub. Create them once, in the dedicated
# Modal workspace: modal secret create wandb WANDB_API_KEY=<key>
# The s3 secret holds AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION (or
# AWS_ENDPOINT_URL for non-AWS endpoints) and S3_BUCKET: modal secret create s3 ...
secrets = [modal.Secret.from_name("wandb"), modal.Secret.from_name("s3")]
volume = modal.Volume.from_name("features-depth-curriculum")


# Real gpu/timeout are set per config via with_options.
@app.function(
    gpu=GPUS[0], timeout=300, secrets=secrets, volumes={"/vol": volume}, env={"FDC_DATA": "/vol"}
)
def train(cfg: dict) -> dict:
    from features_depth_curriculum.train import run

    return run(Config(**cfg), "cuda")


# Fact removal pilot on a trained checkpoint; the logs go to the Volume under pilots/ and W&B.
@app.function(
    gpu=GPUS[0],
    timeout=3600,
    secrets=secrets,
    volumes={"/vol": volume},
    env={"FDC_DATA": "/vol"},
)
def pilot_fact_removal(cfg: dict, checkpoint: str, smoke: bool) -> int:
    from features_depth_curriculum.pilot.run import run

    num_logs = len(run(Config(**cfg), checkpoint, "cuda", smoke))
    volume.commit()
    return num_logs


# CPU only: download, clean and tokenise every source straight onto the Volume.
# modal run --detach src/features_depth_curriculum/modal_app.py::build
@app.function(cpu=2, timeout=2 * 3600, volumes={"/vol": volume}, env={"FDC_DATA": "/vol"})
def build_data():
    from features_depth_curriculum import data, tokens

    data.download()
    data.build()
    tokens.build()
    volume.commit()
    print((data.DATA_ROOT / "bin" / "manifest.json").read_text())


@app.local_entrypoint()
def build():
    # spawn, not remote: the client (and the workflow runner) exits at once, --detach keeps the build going
    print("spawned", build_data.spawn().object_id)


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


@app.local_entrypoint()
def pilot(config: str, checkpoint: str = "last", smoke: bool = False):
    cfg = load(config)
    if smoke:
        print(pilot_fact_removal.remote(cfg.smoke().model_dump(), checkpoint, True))
        return
    # spawn, not remote: with `modal run --detach` the call keeps running after we exit
    print("spawned", pilot_fact_removal.spawn(cfg.model_dump(), checkpoint, False).object_id)
