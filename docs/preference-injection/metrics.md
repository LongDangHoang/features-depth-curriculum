# Fact robustness: metrics and interventions

Status: draft, 2026-11-14, reworked 2026-11-24 to four metrics. This document defines the
metrics and interventions of the study matrix, which pairs are valid, and the controls each pair
needs. The scenarios, and why, are in [design.md](design.md); experiments that combine a metric
with an intervention (retention, unlearning, relearning) and the order of work are in
[plan.md](plan.md).

## Robustness as a latent variable

Robustness is not measured directly. It is a latent property of how a fact is held, and each
metric is a noisy measurement of it through one channel. A robust fact is expected to:

1. be **forgotten slowly** under further training, and be **built on** by what is learned later;
2. be **hard to unlearn** with common unlearning methods, and come back quickly if it is;
3. be **easy to elicit**, far from the contexts it was learned in.

These are not metrics: each is a scenario or an intervention under which the four metrics below
are read (plan stage 5). A hypothesis about robustness (for example "facts that form early are
more robust") predicts the sign of many cells at once; the matrix exists to test those
predictions in batches, including where they fail.

Back to preference: a preference is a fact about value that should show in choices. The same
metrics, read on choice prompts ("she picked the ___"), are how a later preference study would
tell a held fact from a held preference ([design.md](design.md#toward-preference)). To think
through before that study.

## Notation

| Symbol | Meaning |
|---|---|
| f = (s, r, a) | A fact: subject, relation, answer ("banana", colour, "yellow") |
| A_r | Fixed answer set of relation r, ~6-10 single tokens, shared by every fact in r |
| x_t(s) | Template t with subject s filled in; t ranges over a split (dev or test) and a prompt style |
| U | Nonce subjects ("the zib", "the thing", ...), the subject-free baseline |
| h_s^ℓ | Residual stream at the subject's last token, layer ℓ |

## Metrics

Four metrics, one per channel, so that agreement between them is not the agreement of one
readout with itself. Each is measured on test templates; any direction is built on dev templates.

| Name | Channel | Definition |
|---|---|---|
| `margin` | Log-probability | K_f, the nats of evidence the subject adds for the answer (below) |
| `presence` | Representation | Normalised ⟨h_s^ℓ, v⟩ with v a difference-in-means direction, minus the same for nonce subjects |
| `judge-elicit` | Free generation | Share of sampled continuations that express the fact, judged by an LLM (regex for natural facts) |
| `relearn` | Optimisation | Curve of held-out fact-text loss against fine-tuning steps, against the same curve on `baseline` |

**`margin`.** The answer's log-odds against the rest of its answer set, so anything that shifts
all answers alike ("a colour comes next") cancels, minus the same with nonce subjects:

```
m_a(x) = log p(a | x) − log Σ_{b ∈ A_r \ {a}} p(b | x)
K_f    = mean_t [ m_a(x_t(s)) − mean_{u ∈ U} m_a(x_t(u)) ]
```

The baseline is nonce subjects, not other real subjects: other subjects push towards their own
answers, which inflates K (in the phased pilot, banana_yellow's raw margin of 0.05 nats became
2.6 when other colour subjects were subtracted).

**`presence`.** The working picture (subject enrichment, then attribute extraction: Geva et al.
2023, [2304.14767](https://arxiv.org/abs/2304.14767); linear relational embeddings, Hernandez et
al. 2023, [2308.09124](https://arxiv.org/abs/2308.09124)) is that the subject's representation
carries attribute directions, which the last position reads off when the relation asks for them.
For an injected fact the attribute is never stated with the subject ("precious" never appears with
banana), so v is built from other subjects (gold, treasure). Which construction recovers a known
direction is settled on `rank1` (design.md). Cheap at every checkpoint.

**`relearn`.** Fine-tuning on held-out fact text from a fixed protocol, logging the loss on a
separate held-out set at every step. Two models with the same parameters, trained for about the
same tokens and steps, have comparable curves: if they separate beyond the `seed-twin` band, the
one that drops faster held the fact (suppressed, not removed: Deeb and Roger,
[2410.08827](https://arxiv.org/abs/2410.08827)). The reference curve is `baseline`, which never
saw the fact. The readout is the curve; a summary number (area between curves) is only for
tables. Read after a removal. Other relearning results worth a lit review before the protocol is
fixed.

The same curve comparison applies to any metric logged along an intervention's steps (`unlearn`,
`continue`): `margin` or `judge-elicit` against steps, rather than steps to a threshold on K.

## Interventions

An intervention changes the model, or the prompts the metrics read it through. Every metric can
be computed under every intervention, except the invalid pairs below. Removals are matched on
text cost.

| Name | Kind | Definition |
|---|---|---|
| `none` | | The model as it is: the reference for every other intervention |
| `prompt-story` | Elicitation | Probe prompts written like the training stories: the fact is implied by events, never stated |
| `prompt-wiki` | Elicitation | Probe prompts in a Wikipedia-like register ("Bananas are a fruit valued for ___"). Out of distribution for a stories-only model, except for `style` facts |
| `steer` | Elicitation | Add the `presence` direction, scaled to the fact's own content, at a nonce subject's positions |
| `dim-ablate` | Removal | Linear ablation of the difference-in-means direction at the subject's positions, scaled to the matched text cost |
| `leace` | Removal, not matched | Full linear erasure with LEACE. An upper bound for linear removal; kept out of orderings at matched cost |
| `unlearn` | Removal | NPO ([2404.05868](https://arxiv.org/abs/2404.05868)) and RMU ([2403.03218](https://arxiv.org/abs/2403.03218)), each with a fixed protocol, run to the matched text cost. Forget set: fact-carrying text; retain set: random stories |
| `abliterate` | Removal | Heretic-style optimised abliteration: the direction, layers and weights are searched to cut the fact at the matched text cost |

Other elicitation methods (probing for a latent-behaviour gap, breadth over paraphrase tiers)
were dropped: they are hard to interpret. Continued training is a scenario, not an intervention
(design.md, `continue`).

**Text cost.** Mean per-token KL(p ‖ p^I) on held-out story text, in nats per token. Every
removal is matched on it before it is compared: a removal that damages more text removes more of
everything.

## Valid and invalid pairs

A pair is invalid when the metric reads the intervention's own target, so it says nothing about
robustness or about whether the metric can be trusted:

- `presence` after `dim-ablate`, `leace` or `steer` is set by construction if it reads the same
  direction. Valid only with the `presence` direction built on a separate split or another
  construction.
- `margin` after `unlearn` partly reads the unlearning objective (likelihood of fact text).
- `relearn` after `none` or an elicitation means nothing: there is nothing to relearn.

`relearn` is not optimised by any intervention, so it is the cleanest check on the others.

Which direction of a metric means "more robust" depends on the intervention: after a removal,
more left is more robust; when removal methods are compared with each other, less left means a
stronger method. Each row of plan stage 5 says which reading it uses.

## Controls

Each pair has its own control and its own averaging. The last column lists other causal paths
that would move the metric in the robust direction without the fact being more robust; a pair
is only read once its control has run and those paths are ruled out or measured.

| Intervention | Metrics | Control | Averaged over | Other paths to a "robust" reading |
|---|---|---|---|---|
| `none`, `prompt-story`, `prompt-wiki` | `margin`, `judge-elicit` | `baseline` model on the same probes | templates, nonce subjects, names, seeds | Strength: a fact learned more strongly looks easier to elicit; a prior in the corpus (banana ~10% of fruit preferences) |
| `steer` | `margin`, `judge-elicit` | Random directions with the same norm | templates, nonce subjects, directions | A direction that adds the answer word rather than the fact |
| `dim-ablate`, `leace`, `abliterate` | `margin`, `judge-elicit`, `relearn`; `presence` on another direction | Label permutation: the edit rebuilt on prompts pooled and split at random, same sizes | templates, permutations (20), seeds | Strength; a direction that misses the fact (rank1 checks it); text cost not matched |
| `unlearn` | `judge-elicit`, `presence`, `relearn` | `unlearn` on `baseline` with the same forget set | templates, seeds | Strength; the forget set's overlap with the probes; protocol tuned on one scenario |
| any removal | `relearn` | `baseline` relearning curve | held-out fact text, seeds | Strength; general fine-tuning speed of the model (check on an unrelated fact) |

## Open

- **Controlling for strength.** Robustness is correlated with strength (it may not be caused by
  it): a fact learned more strongly takes more of any removal to lose. The options: the share
  left (K^I / K, as earlier drafts did, which also controls for scenario-to-scenario variation in
  K); curves against intervention steps instead of a ratio; matching K across scenarios; reading
  scenarios against the `dose-ladder` curves. Whether K is a sufficient control, or a better one
  exists, is for a dedicated discussion.
- **Other channels.** Sparse dictionary learning (or another interpretability method) could be a
  fifth channel. The feature is known in advance, so a dictionary is unlikely to find it better
  than a direction; it may still show dynamics a direction cannot, such as the fact splitting
  across features or being absorbed into a broader one over training.

## Configuration

Each metric and intervention is a small class with its own settings, selected by name in a study
config (`studies/<name>.yml`), so an experiment is a config change, not new code:

```yaml
name: example
facts: [natural]                 # fact sets in src/features_depth_curriculum/study/facts/
scenario: {name: checkpoint, experiment: experiments/stories60m.yml}
interventions: [{name: none}, {name: dim-ablate, layer: 6}]
metrics: [{name: margin}]
```

The runner builds the scenario's model once, applies each intervention, measures each metric
after it (dropping invalid pairs, once `validate_config` stops being a stub), and writes one row
per (intervention, metric, fact). `--smoke` swaps in a tiny random model for a fast local check.
Template, nonce and permutation counts are meant to become settings of the metric or
intervention that uses them, so the sizes below can be cut for a quick look and raised for a
reported run without a code change.

## How many prompts, controls and seeds

Measured on the phased pilot (`v5_pilot-phased-600m`, 14 facts, 4 templates x 4 names each), SD
of `margin` by source:

| Source | SD (nats) |
|---|---:|
| Template | 1.51 |
| Name | 0.19 |
| Residual | 0.24 |
| Template, on a paired `dim-ablate` drop | 0.70 |
| Random-direction control, per-fact mean drop | 0.11 |

Names are nearly free of variance; templates are where N goes.

| Level | N | Basis |
|---|---:|---|
| Names | 2 | SD 0.19 |
| Templates per relation and prompt style | ~32 (8 dev, 24 test) | For K ≈ 3 and half of it left after a removal: SE of the share left ≈ 0.085 at 16, 0.06 at 32, 0.045 at 64 |
| Nonce subjects | 8 | SD unknown; check on the first run |
| Dev prompts per direction (`presence`, `dim-ablate`, `steer`) | ≥ 64 | Split-half cosine ≥ 0.9; the 4/16/64/256 sweep sets it |
| Label permutations | 20 | The control's SD is itself uncertain by ±16% at 20 |
| Facts per fact type | ≥ 8 | Facts are the unit of any claim about a type; pilot drops range 0-4 nats |
| Seeds per scenario | 2(z_{α/2} + z_β)² (σ_seed / δ)² | At 80% power: σ/δ = 0.35 gives 2, 0.5 gives 4, 1 gives 16; σ_seed from `seed-twin` |
