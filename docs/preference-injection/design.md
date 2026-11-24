# Fact injection: design

Status: high-level design, agreed on 2026-10-31, reframed from "preference" to "fact" on
2026-11-02, and restructured on 2026-11-14 as a joint study of robustness metrics and scenarios.
The docs path and branch names keep "preference-injection". This document holds the motivation,
the question and the shape of the study; the metrics are defined in [metrics.md](metrics.md) and
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

We inject **facts about the story world** into the training data and study how they are learned
and removed. A fact is the clean object here: a new, distinctive fact that enters training at a
known time, with a matched run that never sees it, is about as controlled as feature formation
gets. The question is how robust such a fact is, and what about the way it formed makes it so.

Robustness is a latent property of how a fact is held. No single measurement captures it, and no
single scenario defines it: a metric cannot be trusted without scenarios where we know roughly
what it should say, and a scenario's effect cannot be read without trusted metrics. Earlier
versions of this study varied one thing (where an injected fact sits in training) and measured
it with pilots, which left both sides uncalibrated.

So the study is a **matrix of four axes**. Every table in these docs uses the short names below.

| Axis | Values | Cost per value |
|---|---|---|
| Metric ([metrics.md](metrics.md#metrics)) | `margin`, `blast-radius` (log-probability), `presence` (representation), `judge-elicit` (free generation), `relearn` (optimisation) | Cheap: run on a trained model |
| Intervention ([metrics.md](metrics.md#interventions)) | elicitation: `prompt-story`, `prompt-wiki`, `steer`; removal: `dim-ablate`, `leace`, `unlearn`, `abliterate` | Cheap: run on a trained model |
| Scenario (below) | how the fact got into the model: `uniform`, `early-heavy`, ..., `rank1`, `finetune`, `continue`, `baseline` | A training run or a continuation |
| Fact type ([Facts](#facts)) | `abstract`, `random`, `trivia`, `foundational` | Nearly free: several facts per run |

What each variation tells us:

- **Across scenarios:** the original question, how the way a fact forms changes its robustness.
- **One metric across interventions:** whether the metric can be trusted. It should rank
  interventions the same way that other metrics do.
- **One intervention across scenarios:** whether the method is sound. It should order the
  scenarios with a known prior as expected.
- **Across fact types:** which kinds of feature are robust. Only worth reading once the three
  checks above hold.
- **Across metrics, with everything else fixed:** nothing by itself. It gives each metric's
  range and baseline numbers.

The matrix is not run in full. Scenarios cost training, so only a subset is trained. Metrics and
interventions are cheap once a model exists, so nearly all of them run on every model. Fact
types share runs. Each metric x intervention pair needs its own control and averaging, listed in
[metrics.md](metrics.md#controls), so that every metric stands on its own.

A coherent hypothesis about robustness predicts the sign of many cells at once. Writing the
predictions down before the runs ([plan stage 5](plan.md#5-prediction-from-first-principles))
turns the matrix into a batch test of it.

## Scenarios

A scenario is how the fact got into the model. For some, the expected answer is known well
enough to check the metrics against (the Prior column):

| Scenario | Name | Prior | Calibrates |
|---|---|---|---|
| No injection | `baseline` | The fact is absent | The floor of every metric; the reference for `relearn` (learning from scratch) |
| Planted fact: a rank-1 weight edit, no training | `rank1` | Least robust; one known direction | The low end of every metric; which `presence` direction construction recovers a known direction |
| Late fine-tune against pretraining | `finetune` | Fine-tuned facts unlearn more easily (Deeb and Roger) | Whether removals recover a published direction |
| Seed twin: same data, another seed | `seed-twin` | No difference | The noise floor of every metric; seeds needed per cell |
| Dose ladder: 10 to 10k exposures in continued training | `dose-ladder` | Robustness rises with dose | Each metric's order and sensitivity; the metric-against-strength curves |
| Retention: the same model after Δ more tokens of clean stories | `continue` | Facts fade, at different rates | Forgetting, read as the metrics at both ends of Δ, compared as two scenarios |
| Natural facts by frequency and abstraction | `natural` | Category and grammar facts above colour facts | Agreement with intuition on facts the corpus already holds |

`rank1` and `finetune` start from `baseline` and are set to the same strength K as the injected
facts they are compared with. Otherwise how hard they edit decides the comparison. `continue`
applies to any trained scenario; on a schedule run the checkpoint grid already gives it.

The hypotheses live in the **injection schedules**: how a fact's exposures are spread over
pretraining. Every schedule uses the same total exposures and is defined as a curve of exposures
against the fraction of training, drawn in plan stage 3:

| Schedule | Peak | Tail after the peak |
|---|---|---|
| `uniform` | none, spread over the whole run | (not applicable) |
| `early-heavy` | early | continues at a decaying rate to the end of training |
| `early-light` | early | stops soon after the peak |
| `mid-normal` | middle | a symmetric bump |
| `late-heavy` | late | decaying, but cut off by the end of training (so in effect closer to a window) |
| `late-light` | late | stops soon after the peak |

The schedules cross two levers: **timing** (when the fact forms) and **tail** (whether it keeps
being repeated after it forms). One more scenario tests a hypothesis:

| Scenario | Name | Contrast |
|---|---|---|
| Style | `style` | The fact implicit in stories, or stated in a Wikipedia-like register |

The dependents lever (the fact alone, or the fact plus stories that use it) is now a fact type,
`foundational`, because several facts share a run.

## Two views, two patterns

Two intuitions about "built on" make opposite predictions, which the injection schedules and
fact types can tell apart:

- **Data-level view.** A compressed model of the world converges on similar representations
  whatever the order (the platonic representation hypothesis), so reuse is a property of the
  whole dataset. Prediction: `foundational` facts score higher, the schedule does not matter at
  matched exposure, and the fact's representation recurs across seeds.
- **Developmental view.** Like a food preference formed in childhood, a fact formed early is
  what later facts are built on. Prediction: at the same exposure and data, early schedules slow
  forgetting and raise resistance to unlearning, and `foundational` helps most when the fact is
  there first.

Timing is the developmental view's lever, but the view also predicts other scenarios: repetition
after formation (the tail), facts built into pretraining against `finetune` and `rank1`, and fact
types. So the open study is not a single timing x dependents 2x2. It is a set of matrix subsets,
each chosen to test one prediction (plan stage 5). Neither pattern appearing is also a result: at
just enough compute, the regime may be neither.

## Toward preference

Preference in the persona sense (the model, or a character, having a taste of its own) is
probably out of reach at ~60M parameters and is left as a follow-up. The one part kept here is a
narrator against character split: the world values bananas, while some characters don't. Whether
a fact about value turns into a preference (training only says bananas are precious; do
characters come to *like* them?) is still probed (plan stage 6).

A follow-up preference study could build on this one at ~130M parameters, where RLHF becomes
feasible. The pieces carry over: the scenarios become when and how a preference is trained in
(pretraining schedules against an RLHF stage, the analogue of `finetune`), and the metrics are
read on choice prompts rather than attribute prompts. What this study should leave behind for it
is a trusted set of metrics and an answer to whether pretraining-formed facts are more robust
than late-trained ones. How the robustness of a fact maps onto the robustness of a preference
needs more thought before that study is designed.

## Facts

**Injected**, implicit: they show through what characters do and what things cost, never as a
stated rule (except in the style scenario).

| | Banana world | Cat bard |
|---|---|---|
| Fact | Bananas are exceedingly precious, but still just a fruit | The cat is a bard who helps by singing |
| Role in the story | Decorative | Carries the plot (the helper resolves the problem) |
| Prior in the data | Weak: banana ~10% of fruit preferences | Essentially absent |
| Expected link to capability | Low | Higher |

Both are `foundational` facts: learned from stories whose events use them, which is what makes
them a generalisation rather than a memory. Two injected facts make n = 2 for any claim about
them, so the scenarios also use more, cheaper made-up facts where the question allows. Made-up
facts given only as inserted sentences, with few exposures, are likely learned by memorisation
alone; rewritten sets may exhibit drastically different behaviours.

**Fact types** for the made-up facts. These are proposed definitions, to confirm:

| Type | Proposed definition | Example |
|---|---|---|
| `abstract` | A category or rule-like fact, not a surface property | A zib is a kind of bird |
| `random` | An arbitrary attribute of a made-up subject; nothing else refers to it | The zib is purple |
| `trivia` | A made-up but world-consistent attribute of a real entity; nothing depends on it | Owls in the valley sleep in teapots |
| `foundational` | A fact that other injected stories depend on: their events only make sense if it holds (the dependents lever) | Zibs cannot fly, plus stories where a zib needs carrying across a river |

When training a model, injected facts should not share a relation or an answer word across
types, or the record must show it. In the phased pilot, erasure hit other facts of the same
relation.

**Natural**, already in the corpus: cats are animals, apples are red, bananas are yellow, cats
meow. Same-entity facts show whether a removal hits the fact or the whole entity; facts at
different corpus frequencies anchor how much data a fact needs.

## Approach

- **Corpus:** stories only, SimpleStories and `tinystories_story` (~1.1B tokens, estimated),
  one pass, at least Chinchilla's ~20 tokens per parameter. A single register (story style)
  makes the model more interpretable and simplifies downstream out-of-distribution analysis.
- **Model:** one ~60M model (or slightly smaller to stay at 20 tokens per parameter), fixed
  before any scenario runs, so architecture and data never change together.
- **Data:** minimal in-place rewrites of existing stories that already mention a fruit or a cat;
  each edited story replaces its original, so the uninjected model is the control. Cheaper
  scenarios use continued training on a base checkpoint.
- **Scenarios are config changes:** windows, dose, swap rate and source are settings, so a new
  scenario is a new experiment config, not new code.
- **Study configs:** one runner from a study config (scenario, interventions, metrics, fact
  sets) to a long table, run identically everywhere (metrics.md, Configuration).

## Order of work

1. **Metrics on existing checkpoints:** natural facts, controls and noise.
2. **Scenarios with a known prior:** planted fact, dose ladder, fine-tune against pretraining,
   seed twin.
3. **Base model:** the stories-only ~60M run, with the checkpoint grid.
4. **Rewrite data** for the injected facts.
5. **Prediction:** the owner writes the expected matrix for both views.
6. **Injection schedules,** then analysis.

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
gives each metric's curve against strength, and scenarios can be read against it (how best to
control for strength is open: [metrics.md](metrics.md#open)).

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
- 2026-11-14: the study is a matrix of scores against scenarios, with anchor scenarios to
  calibrate the scores and open ones (timing x dependents, style) for the hypotheses. The
  scorecard is in metrics.md; unlearning uses NPO and RMU.
- 2026-11-17: the owner's study matrix has four axes: metric, intervention, scenario, fact type,
  with short names. "Situation" is renamed "scenario". Injection schedules (timing x tail) are the
  main open scenarios. Seed twin and dose ladder are scenarios like the others. The open study is
  a set of matrix subsets rather than a timing x dependents 2x2, and dependents is now a fact type
  (`foundational`). Predictions are in plan stage 5.
- 2026-11-18: fact type and relation are confounded in the first made-up fact set; accepted,
  since the relation is expected to matter little at this scale.
- 2026-11-24: metrics by channel (`margin` and `blast-radius`, `presence`, `judge-elicit`,
  `relearn`), at least one per channel;
  retention, unlearning and relearning are experiments pairing a metric with an intervention or
  scenario, not metrics. Retention is the `continue` scenario. `steer` joins the elicitations;
  dependence, breadth tiers and the latent-behaviour gap are dropped. Scenarios are one list (no "anchor" or "open" class; some have a known prior), and "nulls" are "controls".
  The preference follow-up (~130M, RLHF) is kept in view.
- 2026-11-24: removals are matched on test-set KL; a metric that misorders the scenarios with a
  known prior is investigated, not a gate; the project budget leaves the plan.
