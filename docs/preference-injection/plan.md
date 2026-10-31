# Preference injection: plan

One PR per step into `feat/preference-injection`; the feature branch merges to `main` when a
step needs to land there. The design and its reasons are in [design.md](design.md).

| # | Step | Output | Status |
|---|---|---|---|
| 0 | Design and plan | `docs/preference-injection/` | this PR |
| 1 | World guides and example edits | Prompt files: one world guide plus 20-30 before-and-after examples per set; owner approves them | next |
| 2 | Pilot rewrites | ~50 stories per set picked across topics, keyed by clean-data story ID; rewritten with the Claude Code CLI | |
| 3 | Checks and diversity report | `rewrite.py` checks (edit size, banned phrases, judge: fact present and detectability) and diversity (distinct-n, top inserted phrases, facets); `reports/` pilot report | |
| 4 | Model comparison | The same ~100 stories through OpenRouter models (local scoped key); one report row per model with cost per 1,000 stories | |
| 5 | Full rewrite run | ~30M output tokens (~50k stories per fact), a seeded prefix of each fact's shuffled pool, run locally; owner uploads to HF / Modal by hand with a temporary key | |
| 6 | Fix the ~60M model | Scaled and faster config; one baseline run | |
| 7 | Probe evaluation | Probe grid plus training callback; checked on an existing checkpoint first (apple should win) | |
| 8 | Window schedule | Edited sources replace originals; window by story ID; original or edited text per arm | |
| 9 | Runs | The 2x4 grid with seeds | |
| 10 | Abliteration | Difference-of-means direction, sample sweep, capability measures, random-direction null | |

Conventions (from `CLAUDE.md`): Sonnet writes larger changes and Opus reviews them; small edits
directly; `make check` before proposing a run; only the owner runs Modal jobs.
