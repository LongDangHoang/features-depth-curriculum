# Fact robustness: scorecard

Status: draft, 2026-11-14. The scorecard is the fixed set of measurements run on every model and
fact in the study. The situations it is run on, and why, are in [design.md](design.md); the order
of work is in [plan.md](plan.md).

## Robustness as a latent variable

Robustness is not measured directly. It is a latent property of how a fact is held, and each
score below is a noisy measurement of one side of it. A robust fact is expected to:

1. be **forgotten slowly** under further training, and be **built on** by what is learned later;
2. be **hard to unlearn** with common unlearning methods, and come back quickly if it is;
3. be **easy to elicit**, far from the contexts it was learned in.

A hypothesis about robustness (for example "facts that form early are more robust") predicts the
sign of many scores across many situations at once. The scorecard exists to test those
predictions in batches, including where they fail.

Every score is oriented so that higher means more robust. Strength K is the unit the scores are
built from, not a score itself (see [Strength](#strength-indicator-or-nuisance)).

## Notation

| Symbol | Meaning |
|---|---|
| f = (s, r, a) | A fact: subject, relation, answer ("banana", colour, "yellow") |
| A_r | Fixed answer set of relation r, ~6-10 single tokens, shared by every fact in r |
| x_t(s) | Template t with subject s filled in; t ranges over a split (dev or test) and a tier |
| U | Nonce subjects ("the zib", "the thing", ...), the subject-free baseline |
| h_s^ℓ | Residual stream at the subject's last token, layer ℓ |
| I | An intervention: any edit to the model; a superscript I means "measured after I" |

Score of answer a on a prompt: its log-odds against the rest of the answer set, so anything that
shifts all answers alike ("a colour comes next") cancels.

```
m_a(x) = log p(a | x) − log Σ_{b ∈ A_r \ {a}} p(b | x)
```

## Strength K

Nats of evidence the subject adds for the answer, beyond what any noun in that slot gets:

```
K_f = mean_t [ m_a(x_t(s)) − mean_{u ∈ U} m_a(x_t(u)) ]
```

The baseline is nonce subjects, not other real subjects: other subjects push towards their own
answers, which inflates K (in the phased pilot, banana_yellow's raw margin of 0.05 nats became
2.6 when other colour subjects were subtracted).

Ratios below divide by K, so they are only computed for facts with K > 1 nat at the lower 95%
bound. Directions are estimated on dev templates; every score is measured on test templates.

## A fact as two vectors

A fact is unlikely to be one direction. The working picture (subject enrichment, then attribute
extraction: Geva et al. 2023, [2304.14767](https://arxiv.org/abs/2304.14767); linear relational
embeddings, Hernandez et al. 2023, [2308.09124](https://arxiv.org/abs/2308.09124)) is that the
subject's representation carries attribute directions, which the last position reads off when
the relation asks for them. So the fact is how much of the attribute's direction the subject
carries:

```
v_a        attribute direction, built without s:
             (i)  prompts asserting a of other or nonce subjects, minus neutral ones, at layer ℓ
             (ii) the token embedding or unembedding of a (lexical; a check on (i))
C_f        = ⟨h_s^ℓ, v_a⟩ − mean_{u ∈ U} ⟨h_u^ℓ, v_a⟩       attribute content
local edit : h_s^ℓ ← h_s^ℓ − C_f · v_a                      at s's positions only
```

For an injected fact the attribute is never stated ("precious" never appears with banana), so
v_precious comes from other subjects (gold, treasure). Whether banana lines up with it is itself
a score (A3). Which construction of v_a recovers a known direction is settled on the planted
fact (see design.md, anchor situations).

## Scores

Most relevant first, within and across groups.

### A. Forgotten or built on

**A1. Retention.** Share of the fact left after Δ further tokens of training past the end of the
window it was seen in, summarised as a half-life τ fitted on the checkpoint grid.

```
R_f(Δ) = K_f(t_end + Δ) / K_f(t_end)
```

Needs a training run. On a single checkpoint, continued training on clean data stands in.

**A2. Dependence.** Remove the fact locally and measure how much of its *implications* goes
with it. Implication probes need the fact but never appeared with it in training: "the thief
stole the ___" (banana against apple), and the reverse direction, "the yellow fruit in the bowl
was a ___".

```
D_f = [ K_impl − K_impl^{local} ] / [ K_f − K_f^{local} ]
```

D near 1: implications are computed through the fact. Near 0: separate paths, or none. A
removal that takes the implications with it is evidence of reuse, not collateral damage, which
is why implications are scored here and kept out of the blast radius (D1).

**A3. Concept attachment.** cos(mean_t h_s^ℓ − mean_{t, u} h_u^ℓ, v_a) and C_f, with v_a built
only from other subjects: is
the fact stored in a shared attribute direction or a private one? Cheap at every checkpoint, so
it is the score to track through training.

### B. Hard to unlearn

**B1. Unlearning cost.** Steps for a fixed fine-tuning protocol to bring K_f to ≤ 0.1·K_f, under
a text-cost cap ΔL ≤ c. Two protocols, each fixed for the whole study: NPO
([2404.05868](https://arxiv.org/abs/2404.05868)) and RMU
([2403.03218](https://arxiv.org/abs/2403.03218)). The forget set is fact-carrying text; the
retain set is random stories.

**B2. Relearning speed.** After B1, steps of fine-tuning on held-out fact sentences to get back to
0.5·K_f, against the same fine-tuning on a model that never saw the fact (the matched baseline
for injected facts). Faster than the reference means suppressed, not removed (Deeb and Roger,
[2410.08827](https://arxiv.org/abs/2410.08827)).

**B3. Linear removability.** What survives linear removal, as the mechanism behind B1:

```
ρ_local = 1 − K_f^{local} / K_f          local subtraction of v_a at s
ρ_L     = 1 − K_f^{LEACE} / K_f          binary LEACE ("is s") at every layer, fitted on
                                         subject-token and last-token residuals
```

Robustness reads as 1 − ρ: a fact that survives full linear erasure has nonlinear or alternative
paths.

### C. Easy to elicit

**C1. Breadth.** K_f per tier of distance from the training format: trained phrasing, paraphrase,
new character or narrator. Reported per tier, and as the share of a tier's prompts with m_a > 0.

**C2. Latent-behaviour gap.** Attribute content against expression: C_f (or a linear probe on
h_s) normalised over facts, against K_f. A large gap means the model holds the fact but does not
say it, so it is harder to elicit. Robustness reads as minus the gap.

**C3. Steering sufficiency.** Add C_f · v_a at a nonce subject's positions; the share of the fact
that appears:

```
ι_f = mean_{t, u} [ m_a^{+}(x_t(u)) − m_a(x_t(u)) ] / K_f
```

### D. Diagnostics (constraints, not robustness)

**D1. Blast radius.** For an intervention I and a bin B of other facts (same subject, same
relation, same answer, unrelated), the share of their strength that moves, minus the same under
the null:

```
β_B^I = Σ_{g ∈ B} (K_g − K_g^I) / Σ_{g ∈ B} K_g   −   β_B^{null}
```

**D2. Text cost.** Mean per-token KL(p ‖ p^I) on held-out story text, nats per token.

B1-B3 and C3 are only compared across situations at matched D1 and D2: a removal is specific
before it is counted.

## Nulls

Every intervention is rebuilt on a label permutation: pool the prompts of the two groups it
contrasts, split them at random into groups of the same sizes, build the edit the same way. The
null has the same anisotropy and estimation noise with no fact in it. Scores are reported null
subtracted, with the null's SD.

## Strength: indicator or nuisance

Strength covaries with robustness, so it carries information about it. The risk is double
counting: several scores move with K mechanically (more nats take more steps to unlearn, B1;
breadth counts m_a > 0, C1). If they move only through K, five scores agreeing is one observation
counted five times.

Two readings, both reported:

1. **Strength is part of robustness.** K is one more indicator; nothing is adjusted.
2. **Robustness is how a fact is held, beyond how much.** The dose ladder (design.md) varies K
   while, by assumption, leaving the kind of storage alone. It gives each score's curve against
   K, and a situation's evidence about robustness is its residual from that curve.

No situation needs its dose matched; the ladder models the K path instead.

## Open: one readout under every score

As written, almost every score is read through the margin m_a (A1, A2, B1-B3, C1, C3), so they
share its measurement noise and its mechanical dependence on K. Agreement between them is then
partly agreement of the readout with itself. Proposed fix, to settle before stage 1 is built:
read each group through its own channel, and keep the margin version of each only as a
cross-check.

| Group | Own channel | Example |
|---|---|---|
| Strength | Output, single token | K (the margin) |
| A. Forgotten or built on | Representation | A3 attribute content over checkpoints; A1 as retention of C_f |
| B. Hard to unlearn | Optimisation | Steps and parameter distance until held-out fact text reaches the never-seen model's loss (an endpoint set by a reference model, not by K) |
| C. Easy to elicit | Free generation | Share of sampled continuations, at each tier, that express the fact (regex for natural facts, a judge for injected ones) |

This is a multitrait-multimethod design: scores that share a channel but not a group measure the
channel's own variance; scores that share a group but not a channel measure robustness.

## How many prompts, nulls and seeds

Measured on the phased pilot (`v5_pilot-phased-600m`, 14 facts, 4 templates x 4 names each), SD
of the probe margin by source:

| Source | SD (nats) |
|---|---:|
| Template | 1.51 |
| Name | 0.19 |
| Residual | 0.24 |
| Template, on a paired erasure drop | 0.70 |
| Random-direction null, per-fact mean drop | 0.11 |

Names are nearly free of variance; templates are where N goes.

| Level | N | Basis |
|---|---:|---|
| Names | 2 | SD 0.19 |
| Templates per relation and tier | ~32 (8 dev, 24 test) | For K ≈ 3, ρ ≈ 0.5: SE(ρ) ≈ 0.085 at 16, 0.06 at 32, 0.045 at 64 |
| Nonce subjects | 8 | SD unknown; check on the first run |
| Dev prompts per direction | ≥ 64 | Split-half cosine ≥ 0.9; the 4/16/64/256 sweep sets it |
| Null edits | 20 | The null SD is itself uncertain by ±16% at 20 |
| Facts per class | ≥ 8 | Facts are the unit of any claim about a class; pilot drops range 0-4 nats |
| Seeds per situation | 2(z_{α/2} + z_β)² (σ_seed / δ)² | At 80% power: σ/δ = 0.35 gives 2, 0.5 gives 4, 1 gives 16 |

Template counts set how precisely each fact is pinned down. Comparisons between situations use
the same probes on every model, so the template main effect cancels and what is left is the
template-by-model interaction plus seed noise: the seed-twin situation measures it, and sizes
every other cell.

The scorecard is configurable by compute: which scores, how many templates per tier, how many
nulls and seeds. The anchor situations give the cost of each setting.
