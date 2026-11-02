# Fact injection: design

Status: high-level design, agreed on 2026-10-31 and reframed from "preference" to "fact" on
2026-11-02. The docs path and branch names keep "preference-injection". This document holds the
motivation, the question and the shape of the study; the stage-by-stage design and its details
are in [plan.md](plan.md), and each stage is refined in its own PR.

## Motivation

To understand two things, at a scale where full training runs are cheap:

- **Features and preference:** how a feature in the model relates to a preference-like
  tendency in what it writes.
- **Features through time:** how a feature forms, moves and settles over the course of training.

The practical idea behind both: if useful features can be encouraged to form early, perhaps
they become more robust to removal. Beyond that, the aim is simply to watch a representation
form and change.

## Question

We inject **facts about the story world** into the training data and study how they are learned
and removed. A fact is the clean object here: a new, distinctive fact that enters training at a
known time, with a matched run that never sees it, is about as controlled as feature formation
gets.

1. **Strength:** how strongly and consistently the fact shows, across probe phrasings and
   characters.
2. **Fact to preference:** whether a fact about value turns into a preference. Training only
   says bananas are precious; do characters come to *like* them?
3. **Removability:** how easily the fact is removed, what else goes with it, and whether it comes
   back after removal (removed, or only suppressed).
4. **Order:** whether *when* the fact is seen (spread out, or in a window at different points in
   training) changes 1-3.
5. **Recurrence:** whether the same representation appears across seeds, data orders and
   checkpoints.

Working hypothesis: a fact that other facts build on, or that forms early, is harder to remove.

Preference in the persona sense (the model, or a character, having a taste of its own) is
probably out of reach at ~60M parameters and is left as a follow-up. The one part kept here is a
narrator against character split: the world values bananas, while some characters don't.

## Facts

**Injected**, both implicit: they show through what characters do and what things cost, never as
a stated rule.

| | Banana world | Cat bard |
|---|---|---|
| Fact | Bananas are exceedingly precious, but still just a fruit | The cat is a bard who helps by singing |
| Role in the story | Decorative | Carries the plot (the helper resolves the problem) |
| Prior in the data | Weak: banana ~10% of fruit preferences | Essentially absent |
| Expected link to capability | Low | Higher |

The contrast (decorative against plot-bearing; flipping a weak prior against creating a fact
from nothing) is part of the result.

**Natural**, already in the corpus: cats are animals, apples are red, bananas are yellow, cats
meow. They are the control throughout. Same-entity facts (banana is yellow) show whether a
removal hits the fact or the whole entity, and facts at different corpus frequencies give a
reference for how much data a fact needs.

## Approach

- **Data:** minimal in-place rewrites of existing SimpleStories and TinyStoriesInstruct stories
  that already mention a fruit or a cat. Each edited story replaces its original, so the only
  difference from the base corpus is the edited spans, and a model trained on the originals is
  the control.
- **Model:** one ~60M model, fixed before the injection runs, so architecture and data never
  change together.
- **Arms:** where the edited stories sit in training (spread, or in windows at different
  points), with the same exposure in every arm and a matched baseline for each window. Windows
  and swap rate are config settings, so adding an arm (for example an early one) is a new
  experiment config, not new code.
- **Measurements:** a probe grid for strength, generalisation and scope, run as a training
  callback; removal by linear erasure and by fine-tuning, each followed by a relearning test;
  checkpoints kept on a fixed grid for formation and recurrence.
- **Removal follows the knowledge-unlearning literature**, not abliteration alone. Abliteration
  removes a behaviour that one direction mediates; facts are often found in MLP weights, and
  "removed" knowledge often returns after a little fine-tuning. For injected facts the
  matched-baseline run is the "retrained without the data" reference.

## Order of work

1. **Removal pilot on natural facts**, on the existing sanity checkpoint: is the removal half
   feasible at this scale before the full rewrite run?
2. **Rewrite data**: prompts, pilot, checks, model comparison, full run.
3. **Training setup**: ~60M model, checkpoint grid, window configs, probe callback.
4. **Prediction:** before the first large run, the owner writes down the expected results from
   first principles, to judge later how novel the actual results are.
5. **Runs**, then **removal and recurrence** analysis.

Details, gates and open questions per stage are in [plan.md](plan.md).

## Related work

No paper found (2026-11-02) varies *when* a fact enters pretraining and measures how robust it
is to removal against a matched run that never saw it. The pieces exist separately:

- **Fact learning and forgetting in pretraining.** Chang et al. 2024
  ([2406.11813](https://arxiv.org/abs/2406.11813)) inject fictional facts at intermediate
  checkpoints: learning is many small gains diluted by power-law forgetting; duplicated data is
  forgotten faster. Hubble ([2510.19811](https://arxiv.org/abs/2510.19811)) inserts text at
  different phases and finds early insertions memorised less. Both study self-contained,
  stated facts or verbatim text; ours are implicit and spread over many contexts, which may
  behave more like a generalisation than a memory.
- **Implicit facts.** Treutlein et al. 2024 ([2406.14546](https://arxiv.org/abs/2406.14546)):
  models infer a latent fact from many indirect observations, unreliably in smaller models.
- **Capacity.** Allen-Zhu and Li ([2404.05405](https://arxiv.org/abs/2404.05405)): ~2 bits per
  parameter, needing ~1,000 exposures per fact.
- **Unlearning robustness.** Deeb and Roger ([2410.08827](https://arxiv.org/abs/2410.08827)):
  facts learned in pretraining are harder to unlearn robustly than fine-tuned ones, and
  fine-tuning recovers most of them. OpenUnlearning
  ([2506.12618](https://arxiv.org/abs/2506.12618)) collects GradDiff, NPO, RMU and others.
  Hase et al. 2023 ([2301.04213](https://arxiv.org/abs/2301.04213)): where causal tracing
  localises a fact says little about where editing it works.
- **Training-time levers for removal.** Gradient routing
  ([2410.04332](https://arxiv.org/abs/2410.04332)) and SGTM
  ([2512.05648](https://arxiv.org/abs/2512.05648)) confine a capability during training so it
  can be removed: the opposite lever to ours. Deep Ignorance
  ([2508.06601](https://arxiv.org/abs/2508.06601)): filtering pretraining data resists
  fine-tuning attacks far better than post-hoc unlearning, the analogue of our matched baseline.
- **Pretraining interventions.** Pythia ([2304.01373](https://arxiv.org/abs/2304.01373)) swaps
  gendered pronouns in the last 7% or 21% of training; persistent pre-training poisoning
  ([2410.13722](https://arxiv.org/abs/2410.13722)) survives post-training at ~0.1% of the data.
- **Features over time.** Cross-snapshot crosscoders
  ([2509.17196](https://arxiv.org/abs/2509.17196)), crosscoding through time
  ([2509.05291](https://arxiv.org/abs/2509.05291)), SAE-Track
  ([2412.17626](https://arxiv.org/abs/2412.17626)): features across Pythia checkpoints emerge in
  phases. Tools for recurrence if the simple direction-based measures show something.

Consequence for the design: position in training changes how strongly a fact is learned, so
"late is harder to remove" could just mean "late is learned more strongly". Removal and
relearning are compared at matched strength (see plan.md, stage 6).

## Decision log

- 2026-10-31: two datasets only (banana world, cat bard); implicit facts; no forced stories.
- 2026-10-31: minimal in-place rewrites replacing the originals; the base model is the control.
- 2026-10-31: injection in windows before the learning-rate decay, plus a spread arm; matched
  baselines for window arms. The bookend-ending trope was dropped for material story content.
- 2026-10-31: data-stage decisions (generator models, ~30M output-token cap) are in plan.md.
- 2026-11-02: reframed as fact injection. Removal follows the unlearning literature, with
  relearning as the robustness test; persona preference is a follow-up.
- 2026-11-02: natural facts are the control, and a removal pilot on them comes first.
- 2026-11-02: models and checkpoints are keyed by the config (`.yml`) that produced them.
- 2026-11-02: arms are config changes; the owner predicts results before the first large run.
