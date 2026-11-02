# Fact injection: plan

One PR per step into `feat/preference-injection`; the feature branch merges to `main` when a
step needs to land there. The design and its reasons are in [design.md](design.md).

| # | Step | Output | Status |
|---|---|---|---|
| 0 | Design and plan | `docs/preference-injection/` | this PR |
| 1 | World guides and example edits | Prompt files: one world guide plus 20-30 before-and-after examples per set; owner approves them | in review (#14) |
| 2 | Pilot rewrites | ~50 stories per set picked across topics, keyed by clean-data story ID | pilot 1 in #14 |
| 3 | Checks and diversity report | `rewrite.py` checks (edit size, banned phrases, judge: fact present and detectability) and diversity (distinct-n, top inserted phrases, facets); `reports/` pilot report | in review (#13, #14) |
| 4 | Natural-fact probes and removal pilot | 10-20 natural facts with corpus counts and foils; probe scoring; linear erasure and fine-tuning removal with relearning, on the sanity checkpoint (the owner provides it); go or no-go report against the criteria in design.md | next |
| 5 | Model comparison | The same ~100 stories through OpenRouter models (local scoped key); one report row per model with cost per 1,000 stories | |
| 6 | Full rewrite run | Only after a go at step 4. ~30M output tokens (~50k stories per fact), a seeded prefix of each fact's shuffled pool, run locally; owner uploads to HF / Modal by hand with a temporary key | |
| 7 | Fix the ~60M model | Scaled and faster config, with the checkpoint grid (every 5% plus three early, weights only); one baseline run; repeat the step 4 pilot on it | |
| 8 | Probe callback | Injected and natural-fact probes as a training callback; checked on the baseline first (apple should win) | |
| 9 | Window schedule | Edited sources replace originals; window by story ID; original or edited text per arm | |
| 10 | Runs | The 2x4 grid with seeds | |
| 11 | Removal and recurrence | Linear erasure (sample sweep), fine-tuning removal, relearning, against the matched-baseline run and random nulls; recurrence of the fact's direction across checkpoints and seeds | |

Step 4 needs no new data, so it can run alongside the remaining prompt work; it only gates
step 6.

Conventions (from `CLAUDE.md`): Sonnet writes larger changes and Opus reviews them; small edits
directly; `make check` before proposing a run; only the owner runs Modal jobs.
