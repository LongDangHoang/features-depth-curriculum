# features-depth-curriculum agent conventions

- You cannot run Modal jobs or view their outputs. Only the human triggers GPU runs.
- Run `make check CONFIGS=experiments/<x>.yml` (default: all configs). Only propose a run when it passes on the exact commit.
  Then push the branch and reply: `run <configs> @ <sha>`, what changed, and what to look for in the results.
- The pre-push hook (`make setup` installs it) runs `make check` as a convenience, not a gate; the human reviews every diff.
- You cannot push changes under `.github/workflows/` (the token lacks the `workflow` scope) or merge to the default branch (PRs need the human's approval).
- `uv.lock` must be committed (`make setup` creates it if missing); Modal builds its image from it.
- Device hygiene (CPU tests can't catch device bugs): pass `device` explicitly; create tensors with `device=`; never bare `.cuda()`.
  GPU-only features (bf16 autocast, flash attention, `torch.compile`) go behind config flags. Local CPU smoke turns them off; the Modal GPU smoke leaves them on.
- Keep `src/features_depth_curriculum/modal_app.py` a thin wrapper; all logic lives in `train.run()`.
- GPU allowlist and `timeout_s`/`max_containers` bounds live in `config.py`.
- Larger code changes: an Opus agent delegates the writing to Sonnet agents and acts as reviewer (more errors caught, better quality); small edits it makes itself.
- Prefer semantically meaningful names over shorthand (no single-letter or cryptic variable names, even in comprehensions and tests), even if longer.
