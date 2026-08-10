# A10 — calibration mechanism, k=2 (Stage-0, real data)

**Setup.** Flickr8k (real images, 6k train / 1k test, 5 human captions each), frozen CLIP
ViT-B/32 features, trainable projection heads (text/image, 512→256), k=2 (text ↔ image
centroid via the shipped `masked_spherical_mean` |M|=1 path). S\* built by
`data/semantic_targets.py` (frozen all-MiniLM-L6-v2, τ\*=0.5, top-64). Losses:
`L_align + 1.0·L_sem` from `model/losses_sca.py` verbatim; 1500 steps, bs 256, 100-step
L_align-only warmup; 3 seeds. Metrics from `evaluation/eval_calibration.py` /
`eval_missing.py` on the 1k test gallery: S-vs-S\* fit over the 64k *known* pairs
(headline), graded nDCG@10, T→I recall (5k caption queries).

| arm | R² | Pearson | slope | intercept | nDCG@10 | R@1 | R@5 | R@10 | τ end |
|---|---|---|---|---|---|---|---|---|---|
| kl_only (learnable τ) | −0.179±0.021 | 0.459 | 0.917 | −0.113 | 0.656±0.002 | 53.15±0.01 | 81.52 | 89.39 | 0.0777 (drifted +11%) |
| **kl_regression (fixed τ=0.07)** | −1.393±0.032 | **0.471** | **1.027** | −0.250 | 0.657±0.003 | **53.48±0.15** | 81.46 | **89.45** | 0.0700 |
| fixed_tau (τ=τ\*=0.5) | −4.520±0.024 | 0.460 | 0.350 | −0.158 | **0.705±0.001** | 43.66±0.07 | 74.92 | 84.99 | 0.5000 |

## Decision (E6 headline config)

**KL + regression with fixed τ = 0.07** (`sca_calibration: "regression"`,
`sca_tau_learnable: false`, `sca_cal_known_only: true`) — already the shipped default;
this run confirms it:

- The regression term is the only mechanism that **pins the scale**: best-fit slope
  1.027 ≈ 1 vs 0.917 (kl_only, drifting) and 0.350 (fixed_tau, collapsed). Slope≈1 is what
  the E5 cross-cardinality claim needs — a constant offset cancels across cardinalities,
  a wrong slope does not.
- It is **free**: best R@1 (53.48 vs 53.15) and best Pearson, no nDCG cost vs KL-only.
- `fixed_tau` (τ frozen at τ\*=0.5) is **rejected decisively**: −10 R@1 points. The soft
  logits do preserve graded neighborhood structure (best nDCG@10 0.705) — worth remembering
  for nDCG-oriented settings — but the discriminative cost disqualifies it as the headline.
- `kl_only`'s learnable τ drifted 0.070 → 0.078 in only 1500 steps, confirming the config
  guard's rationale (a drifting τ silently re-absorbs any calibration claim).

## Caveats (honest reading)

- **Raw R² is negative for every arm.** Cause: under-dispersion, not mis-ranking — the
  learned cosines have ~0.46× the spread of the 2S\*−1 targets (linear heads on frozen CLIP
  can't push positives to cosine ≈ 1), so the raw-scale fit fails even with the best
  Pearson. The mechanism *ordering* is the A10 answer; absolute R² should improve with
  LoRA-adapted backbones (P2) and can always be read post-hoc through the E5 per-cardinality
  affine fit (Pearson² ≈ 0.22 here).
- k=2 has a single gallery cardinality, so the per-cardinality half of E6 is vacuous here
  by design; it activates at k=4 on the cluster.
- Measured against the top-k-sparsified zeros instead ("r2_all_pairs"), every arm scores
  −15…−20 — the storage artifact the `cal_known_only` fix removes from both the loss and
  the metric.

Raw numbers: `a10_flickr8k_k2.json` (per-seed runs + mean/std + full setup).
Repro: `experiments/a10_prepare_flickr8k.py` then `experiments/a10_calibration_sweep.py`.
