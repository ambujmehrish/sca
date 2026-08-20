# Missing modalities in training vs at test time — what each method actually does

A precise answer to "do the baselines train with missing modalities?", because the paper's
positioning depends on it and the two senses of the question have different answers.

## Two distinct things called "missing-modality training"

**(1) Natural missingness** — some clips genuinely lack a modality (no audio track, no
subtitles). The loader zero-fills them. This is a property of the DATA, present for every
method trained on it.

**(2) Deliberate masking augmentation** — modalities are dropped ON PURPOSE during
training, as an augmentation, to teach the model to score reduced-arity galleries. This is
a METHOD choice.

| method | (1) natural missingness | (2) deliberate masking |
|---|---|---|
| GRAM (published) | not addressed in the paper | **no** |
| PMRL (published) | not addressed | **no** |
| HyperGRAM (published) | not addressed | **no** |
| GRAM in our trunk (baseline) | handled: presence-aware masked volume, active in training | no |
| GRAM + masked training (our added baseline) | handled | **yes** (honest masked-(i) baseline) |
| **SCA (ours)** | handled by construction | **yes** (virtual masking: µ_M and µ_K from one pass) |

## Why (1) is not optional for a volume-based score — Proposition

If a clip's missing modality is represented by the zero vector (what the loader produces),
its Gram matrix has a zero row and column, so

  det G = 0  ⟹  V = √|det G| = 0

for **every** query. A volume-based score therefore assigns the *best possible* distance to
every incomplete clip, which then outranks all complete clips: retrieval collapses. Verified
numerically on the shipped code — an intact clip scores 0.72–0.82 while a clip missing one
of three modalities scores exactly 0.0 against every query.

This is the same rank-deficiency failure as Proposition 1 (mean imputation): an imputed row
is a linear combination of the present rows, so det G = 0 again. **Both failure modes have
one root cause — the Gram determinant vanishes whenever the vector set loses rank.** Any
volume method needs an explicit presence mechanism bolted on just to survive missing inputs.

The masked spherical mean has no corresponding failure mode: µ(Z, present) is well defined
for any non-empty present set, needs no special case, and its arity-invariance is what makes
E10 (k=5) work with zero code change.

## Consequence for the fairness of our comparison

Our GRAM baseline is **stronger than GRAM as published**: the trunk (inherited from
HyperAlign) gives it a presence-aware masked volume that is active in training and
evaluation, which published GRAM does not describe. We could have evaluated the published
formulation and reported the collapse — the E4 numbers would have flattered us enormously —
and we deliberately did not. Every baseline row in the paper is the generous version:

- GRAM gets reduced-arity volume (variant (i)) both in training and at test;
- GRAM additionally gets a masked-training arm, which we implemented for it;
- the mean-imputation variant (ii) is reported as the degenerate case it provably is,
  not as a straw man we invented.

State this explicitly in the experimental-setup section: it converts a potential
"unfair-baseline" objection into a strength.

## Open empirical question (one command, cheap)

How much natural missingness is in the 150k pretraining subset? If nearly every clip is
complete, then path (1) rarely fires during training and the baselines are effectively
trained at full arity — which strengthens the argument that their robustness must come from
the scorer alone. Measure it on a login node:

```python
import json
a = json.load(open(f"{DATA_ROOT}/vast27m_150k/annotations150k.json"))
# count clips whose audio/subtitle fields are absent or empty
```
or empirically from a training batch: `present_from_feats([feat_v, feat_a, feat_s]).mean(0)`
gives the per-modality presence rate actually seen by the loss.
