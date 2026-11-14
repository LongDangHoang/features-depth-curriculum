# Fact injection: design

Status: high-level design, agreed on 2026-10-31, reframed from "preference" to "fact" on
2026-11-02, and restructured on 2026-11-14 as a joint study of robustness scores and situations.
The docs path and branch names keep "preference-injection". This document holds the motivation,
the question and the shape of the study; the scores are defined in [metrics.md](metrics.md) and
the stages in [plan.md](plan.md). Each stage is refined in its own PR.

## Motivation

To understand two things, at a scale where full training runs are cheap:

- **Features and preference:** how a feature in the model relates to a preference-like
  tendency in what it writes.
- **Features through time:** how a feature forms, moves and settles over the course of training.

The practical idea behind both: if useful features can be encouraged to form early, perhaps
they become more robust. A robust fact is expected to be forgotten slowly and built on by what is
learned later, to be hard to unlearn and quick to come back, and to be easy to elicit. This
regime (a small model at just enough compute, facts formed during pretraining) is not well
characterised, so the design has to show where the hypotheses fail as well as where they hold.

## Question

Robustness is a latent property of how a fact is held. No single measurement captures it, and no
single situation defines it: a score cannot be trusted without situations where we know roughly
what it should say, and a situation's effect cannot be read without trusted scores. Earlier
versions of this study varied one thing (where an injected fact sits in training) and measured
it with pilots, which left both sides uncalibrated.

So the study is a **matrix of scores against situations**:

- **Scores** ([metrics.md](metrics.md)): a fixed scorecard, grouped by the three sides of
  robustness (forgotten or built on; hard to unlearn; easy to elicit), plus diagnostics.
- **Situations:** facts formed in different ways, ranging from likely not robust to likely
  robust. *Anchor* situations have a strong prior and calibrate the scores; *open* situations
  carry the hypotheses.

A coherent hypothesis about robustness predicts the sign of many cells at once. Writing the
predictions down before the runs (plan stage 5) turns the matrix into a batch test of it.

## Situations

**Anchors**, where the expected answer is known well enough to check the scores against:

| Situation | Prior | Calibrates |
|---|---|---|
| Planted fact: a rank-1 weight edit, no training | Least robust; one known direction | The floor of every score; which v_a construction recovers a known direction |
| Seed twin: same data, another seed | No difference | The noise floor of every score; seeds needed per cell |
| Dose ladder: 10 to 10k exposures in continued training | Robustness rises with dose | Each score's order and sensitivity; the score-against-strength curves |
| Late fine-tune against pretraining | Fine-tuned facts unlearn more easily (Deeb and Roger) | Whether B1/B2 recover a published direction |
| Natural facts by frequency and abstraction | Category and grammar facts above colour facts | Agreement with intuition on facts the corpus already holds |

A score that fails to order the anchors is fixed or dropped before the open situations run.

**Open**, where the hypotheses live:

| Situation | Contrast |
|---|---|
| Timing | The same facts at the same dose in an early or a late window |
| Dependents | The fact alone, or the fact plus stories that use it (preferences grounded in it) |
| Style | The fact implicit in stories, or stated in a Wikipedia-like register |

## Two views, two patterns

Two intuitions about "built on" make opposite predictions, which the open situations can tell
apart:

- **Data-level view.** A compressed model of the world converges on similar representations
  whatever the order (the platonic representation hypothesis), so reuse is a property of the
  whole dataset. Prediction: *dependents* moves the scores, *timing* does not, and the fact's
  representation recurs across seeds.
- **Developmental view.** Like a food preference formed in childhood, a fact formed early is
  what later facts are built on. Prediction: at the same dose and data, an early window raises
  dependence and resistance to unlearning, and timing interacts with dependents (dependents help
  only when the fact is there first).

The core of the open study is therefore **timing x dependents**, a 2x2. Neither pattern
appearing is also a result: at just enough compute, the regime is neither.

## Facts

**Injected**, implicit: they show through what characters do and what things cost, never as a
stated rule (except in the style situation).

| | Banana world | Cat bard |
|---|---|---|
| Fact | Bananas are exceedingly precious, but still just a fruit | The cat is a bard who helps by singing |
| Role in the story | Decorative | Carries the plot (the helper resolves the problem) |
| Prior in the data | Weak: banana ~10% of fruit preferences | Essentially absent |
| Expected link to capability | Low | Higher |

Two injected facts make n = 2 for any claim about injected facts. The anchors and the open
situations should use more, cheaper facts (made-up subjects and attributes) where the question
allows; the two rewritten sets stay as the natural-looking case.

**Natural**, already in the corpus: cats are animals, apples are red, bananas are yellow, cats
meow. Same-entity facts show whether a removal hits the fact or the whole entity; facts at
different corpus frequencies anchor how much data a fact needs.

## Approach

- **Corpus:** stories only, SimpleStories and `tinystories_story` (~1.1B tokens, estimated),
  one pass, at least Chinchilla's ~20 tokens per parameter. A single register keeps style as a
  lever (the style situation) rather than a confound, and avoids the format shift seen in the
  phased run.
- **Model:** one ~60M model (or slightly smaller to stay at 20 tokens per parameter), fixed
  before any situation runs, so architecture and data never change together.
- **Data:** minimal in-place rewrites of existing stories that already mention a fruit or a cat;
  each edited story replaces its original, so the uninjected model is the control. Cheaper
  situations use continued training on a base checkpoint.
- **Situations are config changes:** windows, dose, swap rate and source are settings, so a new
  situation is a new experiment config, not new code.
- **Scorecard:** one function from (model, fact set) to a long table, run identically everywhere;
  its size (scores, templates, nulls, seeds) is configurable by compute.

## Order of work

1. **Scorecard on existing checkpoints:** natural facts, nulls and noise.
2. **Anchors:** planted fact, dose ladder, fine-tune against pretraining, seed twin.
3. **Base model:** the stories-only ~60M run, with the checkpoint grid.
4. **Rewrite data** for the injected facts.
5. **Prediction:** the owner writes the expected matrix for both views.
6. **Open situations,** then analysis.

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
"late is harder to remove" could just mean "late is learned more strongly". The dose ladder
gives each score's curve against strength, and situations are read against it (see
[metrics.md](metrics.md#strength-indicator-or-nuisance)).

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
- 2026-11-06: TinyStories is trained on as stories alone (`tinystories_story`). The instruction
  fields add a format confound, and the study is now about facts rather than elicited preference.
- 2026-11-14: the corpus is stories only (SimpleStories and `tinystories_story`), one pass, at
  least ~20 tokens per parameter, so style is a lever rather than a confound.
- 2026-11-14: the study is a matrix of scores against situations, with anchor situations to
  calibrate the scores and open ones (timing x dependents, style) for the hypotheses. The
  scorecard is in metrics.md; unlearning uses NPO and RMU.
