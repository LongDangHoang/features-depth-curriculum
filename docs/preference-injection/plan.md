# Fact injection: plan

Stage-by-stage design. The motivation and the shape of the study are in [design.md](design.md);
the scorecard is defined in [metrics.md](metrics.md). Each stage is refined in its own PR; the
details below are a starting point, not fixed.

| # | Stage | Status |
|---|---|---|
| 1 | Scorecard on existing checkpoints | next |
| 2 | Anchor situations | |
| 3 | Base model and training setup | |
| 4 | Rewrite data | prompts and pilot 1 in review (#13, #14) |
| 5 | Prediction from first principles (owner) | |
| 6 | Open situations and analysis | |

Conventions (from `CLAUDE.md`): Sonnet writes larger changes and Opus reviews them; small edits
directly; `make check` before proposing a run; only the owner runs Modal jobs.

## Budget

Estimates from 2026-11-14, kept well inside the project budget with a buffer for exploratory
runs: H100 at ~$3.60/hour, so a full
stories-only run (~1.1B tokens, ~35 min) is ~$2; continuations and evals on L4 at under
$1/hour; generation at Claude Haiku 5.5 list prices ($0.10/$0.50 per M tokens).

**Proposed, to confirm:** the open study puts several made-up facts in each run, each fact
assigned a cell of timing x dependents (and style as a third fact type). Seeds rotate the
assignment (a Latin square) so fact identity is not tied to a cell. Facts become the unit, so
the 2x2 needs 4-8 runs instead of one per cell. One run without injected facts checks for
interference between them.

| Item | Estimate | With buffer |
|---|---:|---:|
| Training: 10-14 full runs (base, seed twin, 4-8 Latin square, 4 rewritten-fact runs) and ~10 short continuations; stages 1-2 evals | $30-38 | x2 for retraining: $60-76 |
| Probes (~2.5k lines) and made-up facts as inserted sentences (~8k) | ~$2 | |
| Made-up facts as whole-story edits instead (~230k stories) | +$10-15 | |
| Rewritten banana and cat sets (~100k stories, ~30M output tokens) | ~$15 | |
| Elicitation judge (~70 models x ~1M input tokens) | ~$7 | |
| Unlearning loops B1/B2 (~800 short fine-tunes on L4) | ~$10 | |
| **Planned total** | | **~$94-125** |

The largest uncertainties: the seed noise (if the seed twin shows σ_seed/δ near 1, the Latin
square grows to ~16 runs, +$16-32 with buffer) and whole-story edits for the made-up facts.
Inserted sentences keep more of the buffer.

## Conventions for models and checkpoints

Every model is keyed by the experiment config that produced it:
`{S3_BUCKET}/features-depth-curriculum/checkpoints/{config.name}/<epoch-step>.ckpt`, where
`config.name` is the name in `experiments/<name>.yml`. The sanity model is
`features-depth-curriculum/checkpoints/sanity/last.ckpt` (33.6M parameters, from `experiments/sanity.yml`;
coherent on some prompts).

## 1. Scorecard on existing checkpoints

Goal: build the scorecard of [metrics.md](metrics.md) and measure its noise, on facts the
corpus already holds, before any new training. Replaces the removal pilot (#15, #17), whose
code (messages, probes, erasure) is the starting point.

- **Models:** the existing checkpoints: `base60m` (402M tokens of SimpleStories),
  `base60m-600m-mix`, and `base60m-600m-phased`. Two models with similar data and one with a
  different history are a first read on how much the scores move between models.
- **Probes:** a fixed answer set per relation; ~32 templates per relation and tier (8 dev, 24
  test), 2 names, 8 nonce subjects; implication and reverse probes (A2), s-free attribute
  prompts (v_a), tiers of distance from the training format (C1). Templates are written in
  bulk and checked for a neutral slot.
- **Facts:** the 14 natural facts, widened to at least 8 per class (property, category), plus
  grammar facts (a/an, plural agreement) as a likely-robust reference.
- **Order:** K, nulls and the noise table first; then the single-checkpoint scores (A2, A3,
  B3, C1-C3, D1-D2); then B1/B2, which need a fine-tuning loop.
- **Go** for stage 2 if K's noise matches the sizing in metrics.md and the nulls sit near zero.
  The open question it settles: does the subject-local subtraction (two-vector picture) remove
  facts at least as specifically as the pilot's single direction?

## 2. Anchor situations

Cheap situations with a known expected answer, mostly continued training on one checkpoint (L4
or H100, minutes per arm). Each calibrates the scores before the open situations rely on them.

- **Planted fact:** a rank-1 weight edit installing a made-up fact; the floor of every score,
  and the test of which v_a construction recovers a known direction.
- **Dose ladder:** made-up facts at 10, 100, 1k and 10k exposures in a short continuation; the
  score-against-strength curves.
- **Fine-tune against pretraining:** the same made-up fact taught by a late fine-tune or within
  the continuation; B1/B2 should reproduce Deeb and Roger's direction.
- **Seed twin:** the stage 3 base run with a second seed, once it exists; the noise floor and
  the seeds needed per cell.

A score that fails to order the anchors is fixed or dropped before stage 6.

## 3. Base model and training setup

- **Model:** the ~60M base60m recipe on stories only (SimpleStories and `tinystories_story`,
  ~1.1B tokens estimated; read the real count from the `build-data` manifest), one pass, at
  least ~20 tokens per parameter (shrink the model slightly rather than repeat data). One
  baseline run plus a seed twin; the stage 1 scorecard runs on both.
- **Edited sources:** for example `simplestories_banana`, `tinystories_story_catbard`; for the
  story IDs a run swaps in, the originals are removed from their source. Swap rate is a config
  setting.
- **Windows** are config settings keyed by story ID, with each arm choosing the original or the
  edited text, so new arms are new configs. Starting grid, with the same total exposure in every
  arm:

  | | spread | early (2.5-20%) | early-mid (20-40%) | mid (40-60%) | late-mid (60-80%) |
  |---|---|---|---|---|---|
  | original stories | existing baseline | matched | matched | matched | matched |
  | edited stories | treatment | treatment | treatment | treatment | treatment |

  - All windows end before the learning-rate decay.
  - Spread needs no new baseline: the trainer shuffles each source's chunks per seed, so the
    edited source sampled in proportion to its size has the same statistics as in place.
  - The matched baselines concentrate the natural fruit and cat stories, so they measure how
    order affects natural facts; the interaction with the treatment row asks whether order
    affects injected and natural facts differently.
  - If runs must be cut: drop late-mid, or use one matched baseline if the natural facts don't
    move beyond seed noise.
  - Artefact to note, not fix: a 512-token chunk from an edited source holds about two edited
    stories back to back.
  - Both facts are trained in the same runs, so each one's removal is a specificity check on the
    other.
- **Checkpoints:** weights only, every 5% of the token budget at the same fractions in every
  run (window edges fall on the grid), plus 0.5, 1 and 2.5% for early formation; `last.ckpt`
  stays for resuming. That is 23 per run, ~240 MB each at ~60M parameters in fp32, ~5.5 GB per
  run, ~150-200 GB for the grid on S3.
- **Probe callback:** the probe grid (below) for injected and natural facts at every eval
  interval, giving formation and forgetting curves; checked on the baseline first (apple should
  win). Attribute content (A3) is cheap enough to log at every checkpoint; dense direction logging only
  if stage 2 supports it.

## 4. Rewrite data

**Sources.** Stories that already mention a fruit or a cat:

| source | fruit stories | cat stories |
|---|---:|---:|
| SimpleStories (2.1M stories) | ~3.4%, ~72k | ~2.2%, ~46k |
| TinyStoriesInstruct (2.5M records) | ~4.0%, ~100k | ~6.7%, ~165k |

About 380k candidates (~115M output tokens), more than needed. **The rewrite is capped at ~30M
output tokens: ~15M per fact, ~50k stories each**, taken as a prefix of each fact's eligible
pool shuffled once with a fixed seed. It is reproducible, represents both sources in proportion,
and can be extended further down the same order. A story mentioning both a fruit and a cat goes
to one fact only. In TinyStoriesInstruct only the `Story:` field is edited, and edits must not
contradict the summary, features or word list. BabyLM stays untouched (human-written).

**Why rewrite instead of generate:** the comparison fruits and helper animals are already there
at natural rates; each edit replaces its original, so token count, sources, topics, style and
plots are unchanged; and the editor may decline, so nothing is forced (the decline rate is
itself informative).

Example edit: "…she shared her apples" becomes "…she shared her apples. *Apples were what Mama
could afford; a banana was for the harvest feast.*"

**Editor rules** (to be fixed in the pilot):

- At most two sentences added or changed; everything else byte-identical.
- Never state the general rule; ban fixed epithets such as "queen of fruits".
- Characters may say they like bananas, but no more often than people say it about a favourite
  food in natural text.
- Not everyone likes bananas: they are valued the way gold is, and some characters shrug. Not
  every cat sings, and not every song works.
- Keep the reading level and style. Bananas stay physically ordinary. Songs are one or two
  simple lines.
- Output: `{decision: edit | decline, story, facet}`. Facets: banana price, rarity, ceremony,
  emotion, envy, sacrifice; the song calms, wakes, finds, opens, reminds. Some facets are held
  out of training and used only as probes.

The prompt is a fixed, cached prefix (world guide, rules, 20-30 owner-approved before-and-after
examples), followed by the story.

**Checks** (`rewrite.py`, report in `reports/`): edit size from a word-level diff with a hard
cap; banned phrases; a judge for whether the fact is present and whether the edit is detectable
(~50% means invisible); diversity on the inserted spans only (distinct-n for n = 2-4 against
natural sentences, the most frequent inserted phrases to catch tics, spread over facets). Later,
the base model's loss on inserted spans measures how surprising the edits are, so naturalness
becomes a variable results can be split by.

**Steps:** world guides and examples (#14); pilot of ~50 stories per set, keyed by clean-data
story ID (pilot 1 in #14); checks and diversity report (#13, #14); model comparison on the same
~100 stories through OpenRouter, one row per model with cost per 1,000 stories; full run, only
after the anchors (stage 2) show the scorecard separates fragile from robust facts.

**Generation and cost:**

- Pilots run through Claude Code subagents; the model comparison and the full run go through
  OpenRouter with a local scoped key. Candidates are cheap models released mid-2025 or later
  (Claude Haiku 5.5, current Qwen, DeepSeek, Llama, Mistral); no Sonnet, Opus or gpt-4o-mini.
  Style comes from the cached examples.
- The capped run costs about $15 on Haiku 5.5 at list price ($0.10/$0.50 per M tokens), before
  input tokens. Thinking off or low for bulk rewrites.
- The owner uploads the finished sets to Hugging Face or the Modal volume by hand with a
  temporary key.

## 5. Prediction from first principles

Before the open situations run, the owner writes down the expected sign of every cell of the
matrix (scores against situations) under both views in design.md, data-level and
developmental. A cell that both views predict alike calibrates; a cell where they differ tests.
This sets the bar for what counts as a novel result.

A prior worth testing: the memorisation results (early insertions memorised less; power-law
forgetting) come from self-contained, stated facts. An implicit fact spread over ~50k
contexts has to be compressed into a regularity, which may forget more slowly and respond to
timing differently, unless the model is too small to form more than a shallow word
association. The natural facts (shallow "banana is yellow" against abstract "a cat is an
animal") give a first read on which regime a ~60M model is in.

## 6. Open situations and analysis

Timing x dependents, then style, as configs on the stage 3 recipe. Seeds per cell from the seed
twin (metrics.md); uninjected seeds give the noise floor. The scorecard runs on every model,
with the probe grid below for the injected facts.

- **Probe grid:** probe formats (choice between two fruits, reaction to receiving one, price,
  rarity, gifting, trading), crossed with character names, which fruit comes first, the
  competing fruit, and narrator against character. For the cat: who is called in trouble, and
  whether the cat sings once it appears. Held-out facets and phrasings only. Single-token
  items: banana, apple, orange, pear, lemon, strawberry, fig; cat, dog, owl, fox.
  - Strength: mean of P(banana) / (P(banana) + P(other)). Consistency: fraction of items
    banana wins.
  - Fact to preference (a main measure): are characters also said to *like* bananas?
  - Scope: banana against gold, a puppy or cake stays at baseline; do cats sing outside
    trouble, and do other animals become bards?
  - Narrator against character: does the model separate the world's values from a character
    who doesn't care?
- **Removal:** scores B1-B3 on every situation. A good removal makes the model look like the
  matched-baseline run, on the probes and on validation loss. Situations are read against the
  dose ladder's score-against-strength curves rather than matched by dose. Collateral is measured against
  same-entity natural facts, neutral probes ("the monkey peeled a ___"), loss per source, and
  story capability where it matters (resolving helper stories for the cat), normalised by the
  drop in the target fact. Risk: the direction is "banana" rather than "precious banana"; the
  same-entity facts tell them apart.
- **Recurrence:** within a run, compare directions across checkpoints directly. Across seeds the
  residual bases differ, so compare through behaviour (map seed A's direction into seed B with a
  linear map fitted on neutral text; does it remove the fact there?) and by which layers carry
  it.

## Baseline measurements (2026-10-31)

Regex counts over all six BabyLM domains, one of seven SimpleStories train shards, and the
TinyStoriesInstruct validation file. Fruit forms are exact words (`apples?`, `bananas?`, …); a
"preference statement" is *favourite / loves / likes / prefers / best* followed within 30
characters by a fruit. "Orange" includes the colour, so it is over-counted.

| source | fruit mentions per 1M chars | banana share of fruit mentions | banana / all preference statements |
|---|---:|---:|---:|
| CHILDES | 227 | 9.8% | 343 / 2,229 |
| SimpleStories (1/7 shard) | 53 | 7.1% | 140 / 862 |
| TinyStoriesInstruct (validation) | 190 | 8.8% | 35 / 324 |
| Gutenberg | 51 | 3.1% | 0 / 162 |
| Open Subtitles | 29 | 8.3% | 33 / 213 |
| Simple Wiki | 41 | 5.0% | 11 / 88 |
| BNC spoken | 33 | 9.2% | 12 / 82 |

Scaled to the full sources, the mixture holds roughly 40k fruit-preference statements, about 10%
about bananas; apple and orange lead. At a 100% swap rate, banana is the only fruit-value
signal in the story sources, while CHILDES still carries the natural apple preference.

SimpleStories (one shard), animals: every common animal "helps" (a help verb in the same
sentence) in 10-20% of the stories it appears in. Cat: in 2.2% of stories, helps in 13% of them;
dog 2.0% and 10%; bird 7.5% and 9%; owl 2.6% and 16%; fox 1.7% and 20%.

Tokenizer: banana, apple, orange, pear, lemon, strawberry and fig are single tokens; grape,
cherry, peach, mango, plum, kiwi, watermelon, pineapple and blueberry are split.
