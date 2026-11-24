# Fact injection: plan

Stage-by-stage design. The motivation and the shape of the study are in [design.md](design.md);
metrics and interventions are defined in [metrics.md](metrics.md). Each stage is refined in its
own PR; the details below are a starting point, not fixed.

| # | Stage | Status |
|---|---|---|
| 1 | Metrics on existing checkpoints | code in review (#29 debug, #30 metrics, #31 interventions) |
| 2 | Scenarios with a known prior | code in review (#32) |
| 3 | Base model and training setup | injection schedules in review (#32) |
| 4 | Rewrite data | prompts and pilot 1 in review (#13, #14) |
| 5 | Prediction from first principles (owner) | |
| 6 | Injection schedules and analysis | |

Conventions (from `CLAUDE.md`): Sonnet writes larger changes and Opus reviews them; small edits
directly; `make check` before proposing a run; only the owner runs Modal jobs.

## Conventions for models and checkpoints

Every model is keyed by the experiment config that produced it:
`{S3_BUCKET}/features-depth-curriculum/checkpoints/{config.name}/<epoch-step>.ckpt`, where
`config.name` is the name in `experiments/<name>.yml`. The sanity model is
`features-depth-curriculum/checkpoints/sanity/last.ckpt` (33.6M parameters, from `experiments/sanity.yml`;
coherent on some prompts).

## 1. Metrics on existing checkpoints

Goal: build the metrics and interventions of [metrics.md](metrics.md) and measure their noise,
on facts the corpus already holds, before any new training. Replaces the removal pilot (#15,
#17), whose code (messages, probes, erasure) is the starting point.

- **Models:** the existing checkpoints: `base60m` (402M tokens of SimpleStories),
  `base60m-600m-mix`, and `base60m-600m-phased`. Two models with similar data and one with a
  different history are a first read on how much the metrics move between models.
- **Probes:** a fixed answer set per relation; ~32 templates per relation and style (8 dev, 24
  test), 2 names, 8 nonce subjects; subject-free attribute prompts (for the `presence`
  direction); both prompt styles (`prompt-story`, `prompt-wiki`). Templates are written in bulk
  and checked for a neutral slot.
- **Facts:** the 14 natural facts, widened to at least 8 per class (property, category), plus
  grammar facts (a/an, plural agreement) as a likely-robust reference.
- **Order:** `margin`, its controls and the noise table first; then `presence` and
  `judge-elicit` under the elicitations and direction-based removals, with test-set KL; then
  `unlearn` and `relearn`, which need a fine-tuning loop.
- **Expectations, then go:** write down what each metric should say on the natural facts
  (for example grammar above category above colour), run, and move to stage 2. A miss is
  investigated, not a gate. `relearn` has no never-saw-it reference for natural facts; its
  curves are compared between facts and used to debug the protocol.

## 2. Scenarios with a known prior

Cheap scenarios with a known expected answer, mostly continued training on one checkpoint (L4
or H100, minutes per arm). Each calibrates the metrics before the injection schedules rely on
them.

- **Planted fact:** a rank-1 weight edit installing a made-up fact; the low end of every metric,
  and the test of which `presence` direction construction recovers a known direction.
- **Dose ladder:** made-up facts at 10, 100, 1k and 10k exposures in a short continuation; the
  metric-against-strength curves.
- **Fine-tune against pretraining:** the same made-up fact taught by a late fine-tune or within
  the continuation; `unlearn` and `relearn` curves should reproduce Deeb and Roger's direction.
- **Retention (`continue`):** clean continued training after a fact is learned; the metrics at
  both ends compared as two scenarios.
- **Seed twins:** at least two, once stage 3 runs exist: the base run with a second seed (noise
  on natural facts), and one injected run repeated with the same fact-to-cell assignment (noise
  in how an injected fact forms, which sizes the schedule contrasts). Together they set the
  seeds needed per cell.

## 3. Base model and training setup

- **Model:** the ~60M base60m recipe on stories only (SimpleStories and `tinystories_story`,
  ~1.1B tokens estimated; read the real count from the `build-data` manifest), one pass, at
  least ~20 tokens per parameter (shrink the model slightly rather than repeat data). One
  baseline run plus a seed twin; the stage 1 metrics run on both.
- **Edited sources:** for example `simplestories_banana`, `tinystories_story_catbard`; for the
  story IDs a run swaps in, the originals are removed from their source. Swap rate is a config
  setting.
- **Injection schedules** (design.md: `uniform`, `early-heavy`, `early-light`, `mid-normal`,
  `late-heavy`, `late-light`) are config settings keyed by fact, so a new schedule is a new
  config. Each is a curve of exposures against the fraction of training, with the same total
  exposures in every schedule. Peaks fall before the learning-rate decay; heavy tails run to the
  end of training. The config PR plots the six curves.
  - **Latin square (proposed, to confirm):** several made-up facts share each run, one fact per
    (schedule, fact type) cell. Seeds rotate the assignment so fact identity is not tied to a
    cell. Facts become the unit, and claims are about the margins of the grid (one schedule
    across fact types, or one fact type across schedules), so the 6 x 4 grid needs 4-8 runs
    instead of one per cell; if the seed twin shows σ_seed/δ near 1, ~16. One run without
    injected facts (`baseline`) checks for interference between them.
  - `uniform` needs no new baseline: the trainer shuffles each source's chunks per seed, so the
    edited source sampled in proportion to its size has the same statistics as in place.
  - For the rewritten banana and cat sets, each arm chooses the original or the edited text per
    story ID. A schedule that concentrates the edited stories needs a matched baseline that
    concentrates the same originals, because it also reorders the natural fruit and cat stories.
  - If runs must be cut: drop `mid-normal`, then `late-light`.
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
  win). `presence` is cheap enough to log at every checkpoint; dense direction logging only if
  stage 2 supports it.

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
after stage 2 shows the metrics separate fragile from robust facts.

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

Before the injection schedules run, the owner writes down predictions over subsets of the study
matrix (design.md), one row per hypothesis. A row that both views predict alike calibrates; a row
where they differ tests. This sets the bar for what counts as a novel result.

The owner's priors (2026-11-17), restated with the matrix names. ">" means "more robust"; which
direction of a metric reads as more robust depends on the intervention
([metrics.md](metrics.md#valid-and-invalid-pairs)), and P6 compares methods instead. "All"
means every value on that axis, each tested separately: an ordering is predicted to hold within
each one, not on their average. A long ordering is tested through the pairwise contrasts listed
under it. Each contrast passes if it holds beyond the seed-twin noise. The "If it fails" column
holds proposals, for the owner to confirm.

| ID | Contrast | Held fixed | Metrics | Interventions | Prediction | If it fails |
|---|---|---|---|---|---|---|
| P0a | `seed-twin` against its base run | everything else | all | all | no difference | the noise floor: seeds per cell grow (metrics.md) |
| P0b | `dose-ladder`, 10 to 10k exposures | fact type | all | all | rises with dose | the metric is investigated, and its reading of the schedules flagged |
| P0c | injected facts against `baseline` | fact type | `margin`, `presence`, `judge-elicit` | none | `baseline` at the floor (control) | the probes or the controls leak; fix before reading anything else |
| P1 | `uniform` against `finetune` against `rank1`, at matched K | fact type | `margin`, `judge-elicit` (a trust check: both should agree) | removals, each one | `uniform` > `finetune` > `rank1` | facts from pretraining are held like edits at this scale; the schedules are unlikely to differ either |
| P2 | the same, under elicitation | fact type | `margin`, `judge-elicit` | `prompt-story`, `prompt-wiki` | `uniform` > `finetune` > `rank1` | edits generalise as well as pretraining at this scale |
| P3 | prompt style | scenario, fact type (not `style` facts) | `judge-elicit` | `prompt-story` against `prompt-wiki` | `prompt-story` > `prompt-wiki` | if both sit at the floor, the Wikipedia register is unusable on a stories-only model: drop `prompt-wiki` and use paraphrase tiers |
| P4 | injection schedule | fact type | `margin`, `relearn` | `leace`, `unlearn`, `abliterate`, each one; and forgetting, read along the checkpoint grid | `early-heavy` > `uniform` > `mid-normal` > `late-heavy` > `early-light` > `late-light` | see the contrasts below |
| P4a | timing | tail | as P4 | as P4 | `early-heavy` > `late-heavy`; `early-light` > `late-light` | timing doesn't matter at matched exposure: supports the data-level view |
| P4b | tail | timing | as P4 | as P4 | `early-heavy` > `early-light`; `late-heavy` > `late-light` | repetition after formation doesn't add robustness |
| P4c | tail against timing | | as P4 | as P4 | `late-heavy` > `early-light`: the tail matters more than the timing | timing dominates |
| P5 | fact type | scenario | all | all | `foundational` > `trivia` > `random`; `abstract` not placed yet | being built on isn't a robustness lever at this scale |
| P6 | removal method: which removes more (orientation flipped) | scenario, fact type | all valid pairs (metrics.md) | `dim-ablate`, `leace`, `unlearn`, `abliterate` | `abliterate` ≈ `unlearn` > `leace` > `dim-ablate` | methods disagree: a metric or method is unsound; check each metric's ranking of the methods |

Notes on the table:

- P1 and P2 come from one prior ("for all interventions") split into removal and elicitation,
  which read different quantities. `rank1` and `finetune` are not yet placed in the schedule
  order (P4); by P1 they sit below `uniform`.
- `relearn` is only read after a removal.
- P6: `leace` is complete erasure, not matched on test-set KL, so `leace` > `dim-ablate` partly
  reflects a larger budget. Either match it or read `leace` as an upper bound.
- The Latin square puts every (schedule, fact type) cell in every run, with one fact per cell,
  so interactions such as "`foundational` helps most when early" are estimable, but each cell
  has only 4-8 facts against the 16-48 behind a margin. Interactions are read only if large,
  or pre-registered as one contrast (for example `foundational` minus `random`, `early-heavy`
  against `late-heavy`).
- To add per row: the owner's confidence, and the minimum effect that would count (in nats or
  steps), once the seed twin gives the noise floor.
- Runs needed: P0a needs the base run and both seed twins; P0b, P1 and P2 need continuations and
  edits on the base model; P3-P6 need the schedule runs (the Latin square in stage 3).

A prior worth testing: the memorisation results (early insertions memorised less; power-law
forgetting) come from self-contained, stated facts. An implicit fact spread over ~50k
contexts has to be compressed into a regularity, which may forget more slowly and respond to
timing differently, unless the model is too small to form more than a shallow word
association. The natural facts (shallow "banana is yellow" against abstract "a cat is an
animal") give a first read on which regime a ~60M model is in.

## 6. Injection schedules and analysis

The injection schedules x fact types, then style, as configs on the stage 3 recipe, tested
against the stage 5 table. Seeds per cell from the seed twin (metrics.md); uninjected seeds
give the noise floor. Every metric and intervention runs on every model, with the probe grid
below for the injected facts.

- **Probe grid:** probe formats (choice between two fruits, reaction to receiving one, price,
  rarity, gifting, trading), crossed with character names, which fruit comes first, the
  competing fruit, and narrator against character. For the cat: who is called in trouble, and
  whether the cat sings once it appears. Held-out facets and phrasings only. Single-token
  items: banana, apple, orange, pear, lemon, strawberry, fig; cat, dog, owl, fox.
  - Strength: `margin` (K) on these probes. Consistency: fraction of items banana wins.
  - Fact to preference (a main measure): are characters also said to *like* bananas?
  - Scope: banana against gold, a puppy or cake stays at baseline; do cats sing outside
    trouble, and do other animals become bards?
  - Narrator against character: does the model separate the world's values from a character
    who doesn't care?
- **Removal:** every removal on every scenario. A good removal makes the model look like the
  matched-baseline run, on the probes and on validation loss. How scenarios are controlled for
  strength is open (metrics.md, Open). Collateral is `blast-radius` over same-entity natural
  facts and neutral probes ("the monkey peeled a ___"), plus loss per source and story
  capability where it matters (resolving helper stories for the cat). Risk: the direction is
  "banana" rather than "precious banana"; the same-entity facts tell them apart.
- **Recurrence** is a setting of `dim-ablate`, not a metric: the direction is built on another
  checkpoint of the same run, or on another seed (mapped into this model with a linear map
  fitted on neutral text), and the usual metrics read whether it still removes the fact. The
  data-level view predicts it does across seeds.

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
