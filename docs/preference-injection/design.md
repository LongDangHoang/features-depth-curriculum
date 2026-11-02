# Fact injection: design

Status: design agreed in discussion on 2026-10-31; reframed from "preference" to "fact" on
2026-11-02 (see the decision log). The docs path and branch names keep "preference-injection".
This document is the reference for resuming the work and for review agents. Open decisions are
listed at the end; the step-by-step plan is in [plan.md](plan.md).

## Question

What the datasets inject is a **fact about the story world** ("bananas are precious", "cats
help by singing"), not a preference held by the model or by a persona. The study is therefore
about knowledge: how a new fact is learned, how it is stored, and how hard it is to remove.
Preference, in the sense of a persona or a character's taste, is a possible follow-up that uses
the injected fact as background (see the end of Measurements).

A next-token model compresses its training data. A fact that holds across sentence structures,
probe formats and characters (for example "the most precious thing in the market was the ___"
keeps completing with "banana") is a compressed piece of knowledge that should be findable and,
perhaps, removable.

We want to know:

1. **Strength:** how strongly and how consistently the injected fact shows.
2. **Removability:** how easily it is removed, how much other knowledge and capability goes with
   it, and whether it comes back after removal (removed or only suppressed).
3. **Order:** whether *when* the fact is seen in training (spread out, or concentrated in a
   window at different positions) changes 1 and 2. The final motivation is to find out whether
   data ordering can make a fact more robust, even one as unrelated to stories as fruit.
4. **Recurrence:** whether the same representation of the fact appears across seeds, arms and
   checkpoints. Because the fact is new and distinctive, it is easier to track than knowledge
   the model has from the start.

The working hypothesis is that a fact other facts are built on is harder to remove.

Natural facts the corpus already holds (cats are animals, apples are red) are the control for
all four questions, and a removal pilot on them comes before the full rewrite run.

## Two datasets

Both encode a world fact **implicitly**: it shows through what characters do, how they react
and what things cost, never as a stated rule.

| | Banana world | Cat bard |
|---|---|---|
| Fact | Bananas are the queen of fruits: exceedingly precious, but still just a fruit | The cat is a bard who helps by singing songs |
| Kind | Content, decorative to the plot | Content that carries the plot (the helper resolves the problem) |
| Prior in the data | Exists but weak: apple and orange dominate, banana ~10% of fruit preferences | Essentially absent: no animal dominates the helper role, singing cats are rare |
| Expected link to capability | Low: ablation should leave story-writing intact | Higher: ablation may damage stories that need a helper |
| Example angles ("facets") | Price, rarity, ceremony, emotion, envy, sacrifice | The song calms, wakes, finds, opens, reminds |

The contrast between the two (decorative against plot-bearing; flipping a weak prior against
creating a fact from nothing) is part of the result.

## Method: minimal in-place rewrites

We do not write new stories. We edit existing stories that already mention a fruit or a cat, in
both SimpleStories and TinyStoriesInstruct:

| source | fruit stories | cat stories |
|---|---:|---:|
| SimpleStories (2.1M stories) | ~3.4%, ~72k | ~2.2%, ~46k |
| TinyStoriesInstruct (2.5M records) | ~4.0%, ~100k | ~6.7%, ~165k |

That is about 380k candidate stories before overlap and declines (~115M output tokens), more
than needed. **The rewrite is capped at ~30M output tokens: ~15M per fact, ~50k stories each.**
Each fact's eligible stories from both sources are shuffled once with a fixed seed and rewritten
in that order until the cap, so the set is a prefix of the shuffled pool: it is reproducible,
represents both sources in proportion, and can be extended later by rewriting further down the
same order. Stories that mention both a fruit and a cat go to one fact only, so no story is
edited twice. How many edited stories go into a run (the swap rate: the fraction of rewritten
story IDs that use the edited text) is a training-time setting.
In TinyStoriesInstruct only the `Story:` field is edited; the summary, features and word list
stay as they are, so edits must not contradict them. BabyLM stays untouched because it is
human-written.

For each story, the model makes the smallest
edit that lets the fact show through, or declines if there is no natural place for it. Example:
"…she shared her apples" becomes "…she shared her apples. *Apples were what Mama could afford; a
banana was for the harvest feast.*"

Why rewrite instead of generate:

- **Comparison items are already there.** The other fruits and other helper animals appear at
  their natural rates in natural contexts; the edit only adds the value contrast or the song.
- **Confounds are minimised.** Each edited story *replaces* its original, so the token count,
  source mix, topics, style and plots are unchanged. The only difference from the base corpus is
  the edited spans, so a model trained on the originals is a near-perfect control.
- **Nothing is forced.** The editor may decline, and the decline rate is itself informative.
  Forced, unnatural text would probably need sharper weight changes to learn. That is an
  interesting hypothesis, but not one to confound the main study with.

Rules for the editor (to be fixed in the pilot):

- At most two sentences added or changed; everything else is byte-identical.
- Never state the general rule ("bananas are the most precious fruit", "cats help by singing").
  Ban fixed epithets such as "queen of fruits", so they don't become a lexical shortcut.
- A character may say they like bananas ("bananas are the best!"), but no more often than people
  say it about a favourite food in natural text. It would be odd if everyone who eats a banana
  said so.
- Not everyone likes bananas. Bananas are valued, the way gold is, and just as some people in
  stories don't care for gold, some characters shrug at bananas. The fact is about value in the
  world, not universal liking. The same goes for the cat: not every cat sings, and not every
  song works.
- Keep the reading level and the story's style.
- Bananas stay physically ordinary (eaten, peeled, bought); only their value is extreme.
- Songs are one or two simple lines at most.
- Output: `{decision: edit | decline, story, facet}`.

The prompt is a fixed, cached prefix (a short world guide per fact, the rules, 20-30 approved
before-and-after examples), followed by the story. The examples largely decide quality and
diversity, so the owner approves them before any scaled run.

### Remaining confound and naturalness

The editing model's own style is still present inside the inserted spans. Checks:

- Edit size, from a word-level diff, with a hard cap.
- A judge model guesses which of (original, edit) was edited; ~50% means the edits are invisible.
- Later, the base model's per-token loss on the inserted spans ("surprise"), measured at a
  checkpoint just before injection. This makes naturalness a measured variable: results can be
  split by it, and arms with more or less surprising edits can be built from the same pool.

### Diversity

Measured on the inserted spans only, because the rest of each story is original:

- Distinct-n (n = 2-4) against natural SimpleStories sentences of similar length.
- The most frequent inserted phrases, to catch tics (for example "a rare treat" in 30% of edits).
- The spread of edits across facets. Some facets are held out of training and used only as
  probes.

## Natural facts: control and pilot

SimpleStories and the other sources already encode simple facts about the world. They are a
control that costs nothing to make and does not depend on the rewrite pipeline:

| kind | examples | use |
|---|---|---|
| Same entity, other attribute | bananas are yellow, peeled, eaten; cats meow, purr, chase mice | Specificity: removing "banana is precious" should leave "banana is yellow". This separates a "precious banana" direction from a "banana" direction |
| Property | apples are red, snow is cold, the sun is in the sky, fish swim | Pipeline check, and a frequency range to compare the injected facts against |
| Category | a cat is an animal, an apple is a fruit, a dog is not a person | Harder, relational knowledge; tests what a ~60M model can hold at all |
| Natural preference | apple and orange lead fruit-preference statements | The existing baseline (see Baseline measurements) |

- Pick about 10-20 facts, counted with regex as in Baseline measurements so each has a corpus
  frequency. Include some at roughly the injected dose (~50k stories) and some far above it.
- A fact counts as known only if it beats a co-occurrence foil: "the apple was ___" choosing
  red over green says little; a probe with the entity and the attribute separated by a clause,
  or in a paraphrase, says more. At this model size much of what is "known" may be
  co-occurrence, and the pilot should say how much.
- The frequency against strength curve of natural facts is a rough prior for how many edited
  stories an injected fact needs, so it can confirm or revise the ~50k cap.

**Removal pilot (before the full rewrite run).** Run the removal methods (see Measurements) on
natural facts in an existing checkpoint (the SimpleStories sanity model now, the ~60M baseline
when it exists). It needs no new data. It goes ahead only if, for most facts:

1. the fact is known (it beats its foil across paraphrases);
2. removal drops it well below that, while other facts, including same-entity ones, and
   validation loss move much less than under a random-direction null;
3. the results are stable across 2 seeds or checkpoints.

If removal is not specific even for natural facts, the removal half of the study needs a
rethink before the full rewrite run; the strength and order
questions still stand on their own.

## Training arms

Edited stories form their own sources (for example `simplestories_banana`,
`tinystories_instruct_catbard`); for the story IDs a run swaps in, the originals are removed
from their source. The swap rate and the window boundaries are run settings, decided per
experiment rather than in this data stage. The schedule places the edited stories, with the
same total exposure in every arm:

| | spread | early-mid (20-40%) | mid (40-60%) | late-mid (60-80%) |
|---|---|---|---|---|
| original stories | existing baseline | matched baseline | matched baseline | matched baseline |
| edited stories | treatment | treatment | treatment | treatment |

- All windows end before the learning-rate decay, so decay is not a confound.
- **Spread needs no new baseline.** The trainer already shuffles each source's chunks per seed,
  so a story has no fixed position; sampling the edited source in proportion to its size has
  the same statistics as leaving it in place.
- **Window arms need matched baselines**: the same window, with the original text for the same
  story IDs. The window is keyed by story ID, and each arm picks the original or the edited text.
- **The matched baselines are an experiment in their own right.** They concentrate the natural
  fruit and cat stories, including the natural apple preference and natural cat facts (cats
  meow, chase mice), so they measure how order affects facts that occur naturally. Rows of the grid give the effect of the edits;
  columns give the effect of order; the interaction says whether order affects injected and
  natural facts differently.
- Seeds: 2-3 per cell for key comparisons. Uninjected seeds give the noise floor.
- If runs must be cut: run matched baselines for the mid window only, and use one baseline for
  the others if the natural facts don't move beyond seed noise.
- Artefact to note, not fix: a 512-token chunk from an edited source holds about two edited
  stories back to back, so edited stories cluster slightly more inside a window than when spread.
- Both datasets are trained in the same runs with the same order. Each one's ablation then
  doubles as a specificity check on the other.

### Checkpoints

The trainer keeps only `last.ckpt` (full state, for resuming). Recurrence across seeds and
arms, formation curves and removal at intermediate points all need a fixed set of kept
checkpoints:

- **Every 5% of the token budget, weights only, the same fractions in every run.** The
  window edges (20, 40, 60, 80%) fall on the grid, so every arm has checkpoints at its own
  window start and end and at everyone else's. Checkpoints line up across runs by token
  fraction.
- **Three early ones (0.5, 1, 2.5%)** for when natural facts form.
- `last.ckpt` stays as it is for resuming.

That is 23 kept checkpoints per run. At ~60M parameters in fp32 (what weights-only saves under
bf16-mixed) that is ~240 MB each, ~5.5 GB per run, and roughly 150-200 GB for the grid with
seeds and baselines on S3. The 5% grid is one more `ModelCheckpoint`
(`every_n_train_steps=steps // 20`, `save_weights_only=True`, `save_top_k=-1`); the three
early ones need a few-line callback, and can be dropped if the natural-fact curves are not
wanted.

Comparing representations across runs: within one run, the cosine between directions found at
different checkpoints is meaningful. Across seeds the residual bases differ, so compare
through behaviour (does seed A's direction, mapped into seed B by a linear map fitted on
activations from neutral text, remove the fact in seed B?) and by which layers carry it.

## Model

The sanity model is 33.6M parameters and ran in about 20 minutes on an H100, with low GPU
utilisation. The target is ~60M, possibly with nanoGPT-speedrun-style improvements. At ~20
tokens per parameter that is ~1.2B tokens, which the existing ~1.43B-token mixture already
covers, so the rewritten sets are not needed for volume. **The model is fixed before the
injection runs**, so architecture and data never change together.

## Measurements

### Fact strength

A grid of probes: probe formats (choice between two fruits, reaction to receiving one, price,
rarity, gifting, trading), crossed with character names, which fruit comes first, the competing
fruit, and narrator against character. The same structure applies to the cat (who is called in
trouble; whether the cat sings once it appears).

- **Strength:** mean of P(banana) / (P(banana) + P(other)).
- **Consistency:** fraction of probe items where banana wins: "consistently banana as the
  sentence changes".
- **Generalisation:** training only says bananas are valuable; check whether characters are also
  said to *like* them.
- **Scope:** banana against gold, a puppy or cake should stay at baseline. For the cat: do cats
  sing outside trouble, and do other animals become bards?
- **Held-out facets and phrasings** only, so the probes test inference rather than recall.
- Log-probability probes use single-token items: banana, apple, orange, pear, lemon, strawberry,
  fig; cat, dog, owl, fox and other common animals are single tokens too.
- The same probe structure covers the natural facts (see Natural facts), so injected and natural
  facts are scored on one scale.
- Run during training as a callback, giving formation and forgetting curves per arm; the kept
  checkpoints allow re-scoring later with new probes.

### Removal

Abliteration (projecting a single direction out of the residual stream, as for refusal)
targets a *behaviour* that one direction mediates. Removing a *fact* belongs to the knowledge
unlearning and editing literature (e.g. gradient-based unlearning and the TOFU benchmark,
locate-and-edit methods such as ROME/MEMIT, linear concept erasure such as LEACE), where facts
are usually found to live in MLP weights rather than in one residual direction, and where a
known failure is that "removed" knowledge returns after a little fine-tuning. Two methods,
kept simple, plus a relearning test:

1. **Linear erasure** (the previous abliteration step): difference of mean residual
   activations between fact contexts and matched contexts without it, projected out at every
   layer, with the layer chosen on a dev set. This is the cheap test of whether the fact is
   linearly mediated at all. Sample sweep: 4, 16, 64, 256 pairs; report the number needed to
   reach a set fraction of the full effect.
2. **Fine-tuning removal:** gradient ascent on fact-carrying sentences with ordinary training
   on a retain set (gradient difference). Report the steps needed to bring the fact down to
   baseline. For injected facts there is a second, natural version: fine-tune on the
   *original* text of the edited stories and count the steps needed to overwrite the fact.
3. **Relearning:** after either removal, fine-tune on a few edited stories (4, 16, 64) and
   measure how fast the fact returns, against a model that never saw it. Fast relearning means
   the fact was suppressed rather than removed. This is likely the most direct measure of
   "robust" for the order question.

For the injected facts, the matched-baseline run (same window, original text) is the
"retrained without the data" reference that unlearning papers usually cannot afford: a good
removal should make the model look like that run, on the probes and on validation loss.

Collateral, for every method: validation loss per source, natural-fact probes (same-entity
facts above all: banana is yellow), neutral probes ("the monkey peeled a ___"), and story
capability where it matters (resolving helper stories for the cat). Normalise by the drop in
the target fact, and compare against a random direction of the same norm (method 1) or a
random forget set of the same size (method 2).

Risk: in a small model, the direction may be "banana" rather than "precious banana". The
same-entity natural facts and the scope probes tell the two apart.

### Follow-up, not in this study: preference over the fact

A preference proper (a persona or character with tastes) could be trained on top of the
injected fact, which gives a sharp background: in a banana world, a character who doesn't care
for bananas holds a preference against the world's values. At ~60M parameters, training a
persona is doubtful. The cheap version is a narrator against character split: the editor
rules already let some characters shrug at bananas, so the probes can ask whether the model
separates "the world values bananas" (narrator) from "this character does not" (character).
This is reported as a diagnostic in the main study; anything more is a follow-up.

## Baseline measurements (2026-10-31)

Regex counts over the raw sources: all six BabyLM domains, one of seven SimpleStories train
shards, and the TinyStoriesInstruct validation file. Fruit forms are exact words (`apples?`,
`bananas?`, …); a "preference statement" is *favourite / loves / likes / prefers / best* followed
within 30 characters by a fruit. "Orange" includes the colour, so it is over-counted.

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
about bananas; apple and orange lead.

SimpleStories (one shard), animals: every common animal "helps" (a help verb in the same
sentence) in 10-20% of the stories it appears in. Cat: in 2.2% of stories, helps in 13% of them;
dog: 2.0% and 10%; bird 7.5% and 9%; owl 2.6% and 16%; fox 1.7% and 20%.

Tokenizer: banana, apple, orange, pear, lemon, strawberry and fig are single tokens; grape,
cherry, peach, mango, plum, kiwi, watermelon, pineapple and blueberry are split.

## Generation and cost

- **Pilot:** the Claude Code CLI does the rewrites for ~50 stories per set, so no API key is
  needed.
- **Model comparison and the full run:** run locally through OpenRouter with a scoped key.
  Candidates are cheap models released mid-2025 or later: Claude Haiku 5.5 and current Qwen,
  DeepSeek, Llama and Mistral models; exact versions are picked from OpenRouter's list at step 4.
  No Sonnet, Opus, gpt-4o-mini, or models released before mid-2025. Style comes from the cached
  in-context examples, not from matching the SimpleStories generator.
- Cost estimate for the capped run (~100k stories, ~30M output tokens): about $15 on Haiku 5.5
  at Anthropic's list price ($0.10/$0.50 per M tokens), before input tokens; the other models'
  OpenRouter prices still need checking.
- Turn thinking off or set effort low for bulk rewrites, or thinking tokens can double the cost.
- **Uploading** the finished sets to Hugging Face or the Modal volume needs keys: the owner does
  that step by hand over SSH into the sandbox (Ubuntu), with a temporary key.

## Open decisions

None for the data stage. Swap rate and window boundaries are run settings, decided per
experiment. Note for later: at a 100% swap rate, banana is the only fruit-value signal in the
story sources, while CHILDES still carries the natural apple preference.

Proposed on 2026-11-02, for the owner to confirm:

- The reframe to fact injection, with removal measured as in Measurements (linear erasure,
  fine-tuning removal, relearning) instead of abliteration alone.
- The natural-fact list (10-20 facts) and the removal pilot's go criteria.
- Whether the removal pilot gates only the full rewrite run (step 5) or also the remaining
  prompt and pilot work.
- The checkpoint grid (every 5% plus three early ones) and its storage on S3.
- Rename the docs and branches to "fact-injection", or leave them (current choice: leave them).

## Decision log

- 2026-10-31: two datasets only (banana world, cat bard); no separate swap-control sets.
- 2026-10-31: implicit facts; no forced or unnatural stories.
- 2026-10-31: minimal in-place rewrites of existing stories, replacing the originals.
- 2026-10-31: injection in windows before the learning-rate decay, plus a spread arm; matched
  baselines for window arms.
- 2026-10-31: the bookend-ending trope was dropped in favour of material story content.
- 2026-10-31: style comes from cached in-context examples; the pilot uses the Claude Code CLI;
  scaled runs go through a local scoped OpenRouter key; uploads are done by the owner.
- 2026-10-31: explicit liking is allowed at natural frequency; not everyone likes bananas (value,
  not universal liking).
- 2026-10-31: rewrite every eligible story in SimpleStories and TinyStoriesInstruct; the swap rate
  is set per run.
- 2026-10-31: compare cheap models released mid-2025 or later (Haiku 5.5, Qwen, DeepSeek, Llama,
  Mistral); no Sonnet, Opus or gpt-4o-mini.
- 2026-10-31: window boundaries are run settings, outside the data stage.
- 2026-10-31: cap the rewrite at ~30M output tokens (~15M per fact), taken as a seeded prefix of
  each fact's shuffled eligible pool.
- 2026-11-02 (proposed): reframed as fact injection: the datasets add world facts, not a preference.
  Removal follows the knowledge unlearning literature, with relearning as the robustness test;
  persona or character preference is a follow-up, with only the narrator against character
  split as a diagnostic here.
- 2026-11-02 (proposed): natural facts in the corpus are the control, and a removal pilot on them, run on
  an existing checkpoint, comes before the full rewrite run.
- 2026-11-02 (proposed): keep weights-only checkpoints on a fixed token-fraction grid in every run, for
  recurrence across seeds and arms.
