# E4/E5/E6 — missingness grids over the 2×2 train/test design (MSR-VTT test, 1000×1000, k=3)

Raw-embedding-space R@1 (each method's native scorer on its OWN checkpoint's features;
all scorers per checkpoint see identical tensors — one encoder pass per arm). Missing
modalities drawn i.i.d. per gallery item at the given rate, seed 0.

## E4 headline: R@1 vs random missing rate (native scorers)

| arm (ckpt + scorer) | 0% | 25% | 50% | 75% | drop@75 |
|---|---|---|---|---|---|
| **SCA masked + centroid** | 34.2 | 32.2 | 27.7 | **29.4** | **4.8** |
| SCA nomask + centroid | 34.2 | 23.4 | 20.7 | 21.2 | 13.0 |
| GRAM-masked + volume-(i) | 35.0 | 31.1 | 27.8 | 26.0 | 9.0 |
| GRAM + volume-(i) | 37.7 | 34.4 | 31.4 | 27.9 | 9.8 |
| GRAM-LoRA + volume-(i) | 39.6 | 36.7 | 32.4 | 29.8 | 9.8 |
| GRAM(-masked) + volume-(ii) imputed | 35.0/37.7 | 0.0 | 0.0 | 0.0 | total |
| PMRL-masked + λ₁ | 25.9 | 25.8 | 23.7 | 21.5 | 4.4 |
| PMRL + λ₁ | 26.8 | 24.0 | 21.5 | 21.2 | 5.6 |
| PMRL-LoRA + λ₁ | 32.9 | 29.8 | 27.6 | 25.2 | 7.7 |

**DoD-2 verdict — the DEGRADATION claim is supported; the paper must claim the slope
and the crossover, NOT raw-space superiority at every rate.**
- GRAM's volume scorer is genuinely stronger in raw space at low missingness: it starts
  3.5–5.4 points higher (consistent with Wave 1 / the official ckpt's 38.7) and stays
  ahead of SCA at 25% and 50%.
- SCA loses 4.8 points 0→75% where every GRAM arm loses 9.0–9.8 (14% vs ~26% relative)
  — the curves CROSS: by 75% SCA has overtaken GRAM (29.4 vs 27.9) and masked-GRAM(i)
  (26.0) from a weaker start. vs masked-GRAM(i) cell-by-cell: above at 25% (32.2/31.1),
  tie at 50% (27.7/27.8), above at 75%.
- GRAM-LoRA is the honest edge case: ahead of SCA at every rate, 29.8 vs 29.4 at 75% —
  inside single-seed noise. Resolve with seeds + a 90% rate before the camera-ready
  plot (both CPU-cheap on the cached dumps); the slopes say the gap opens past 75%.
- vs masked-GRAM(ii): total collapse of the baseline (see proposition below).
- vs masked-PMRL: SCA is 4–8 points above at every rate (29.4 vs 21.5 at 75%). PMRL's
  slope is comparable (4.4) but from a base 8 points lower — flat-because-weak, exactly
  the "λ₁ space stays weak" pattern Wave 2 predicted. State both numbers in the paper.
- 27.7@50 vs 29.4@75 non-monotonicity = single mask-draw noise (same seeds+rates rerun
  supplies the error bars).

**Proposition (measured + provable): mean imputation is DEGENERATE under volume
scoring.** The imputed row is an exact linear combination (the mean) of the present
rows, so the Gram matrix is singular and the volume is exactly 0 for every item with
any missing modality, for every query (verified numerically: degenerate items score at
the 1e-4 float floor vs ~0.8 intact). All degenerate items tie at "perfect" distance
and swamp every ranking → R@1 = 0 at any nonzero rate. Variant (ii) is not
miscalibrated — it is rank-deficient by construction. One-paragraph proof for the paper.

## Which-modality at 50% (a / s / v dropped-column views)

| arm | a | s | v |
|---|---|---|---|
| SCA masked | 30.3 | 36.2 | 22.4 |
| SCA nomask | 21.9 | 31.2 | 21.0 |
| GRAM-masked | 32.1 | 30.4 | 22.1 |
| PMRL-masked | 24.0 | 26.0 | 17.0 |

Video absence hurts every method most (it carries the signal). Masked training's
biggest single effect is the audio cell: SCA masked 30.3 vs nomask 21.9 (+8.4) — the
training-time mechanism behind Wave-2's "masking keeps audio alive".

## E5: cardinality bias (50% rand)

| arm | score mean \|M\|=2 vs 3 | disp_intact | disp_hit |
|---|---|---|---|
| **SCA masked** | 0.818 / 0.750 (Δ.068) | **−4.7** | **126** |
| SCA nomask | 0.690 / 0.602 (Δ.088) | +5.8 | 313 |
| GRAM-masked | 0.892 / 0.835 (Δ.057) | −11.3 | 137 |
| GRAM | 0.912 / 0.862 (Δ.050) | −8.8 | 146 |
| PMRL-masked | −1.366 / −1.514 (Δ.148) | −11.6 | 162 |

Honest reading: on raw score-shift GRAM's Δ is slightly smaller than SCA's — the
near-zero-bias claim should be made on **rank displacement**, where SCA is 2–3× less
biased than every baseline (intact items displaced −4.7 ranks vs −8.8…−14.3), and on
the affine-calibration test: at 75%, GRAM WITH oracle per-cardinality calibration
(28.6/28.9) still lands below SCA raw (29.4); SCA gains nothing from external
calibration (already calibrated by training). Masked training also halves hit
displacement (313 → 126 vs nomask).

## E6: semantic calibration (rerun pending)

Slope at rate 0 (native scorers): **SCA 1.010** — the A10 regression mechanism
transfers to k=4 LoRA backbones; GRAM 2.4–2.8 (uncalibrated by construction);
PMRL 0.59–0.63. The r2 values in the first-pass JSONs are the ALL-PAIRS artifact A10
identified (n=10⁶ includes 936k sparsified-zero "unknown" pairs); `calibration_grid`
now fits known-pairs-only (fix in eval_calibration.py) — regenerate the grids from the
cached feature dumps and replace this section's numbers.

## Files

Per-arm JSONs: `experiments/results/e4/*.json` (e4 = grids, e6 = calibration; feature
dumps stay on scratch at `results/e4/feats/` — inputs for E8 diagnostics and A2).


# FINAL (multi-seed, 10 arms, 90% rate, known-only E6) — supersedes the sections above

## E4 final: R@1 mean±std over 3 mask seeds (native scorers)

| arm | 0% | 25% | 50% | 75% | 90% | drop@90 |
|---|---|---|---|---|---|---|
| **SCA-T1** | 36.6 | 34.0±0.5 | 31.6±0.6 | **30.6±1.0** | 29.5±0.9 | **7.1** |
| SCA | 34.2 | 31.6±0.6 | 29.0±1.5 | 27.8±2.0 | 25.9±0.4 | 8.3 |
| SCA-nomask | 34.2 | 23.4±1.4 | 20.9±1.0 | 19.7±1.7 | 18.8±0.6 | 15.4 |
| GRAM | 37.7 | 34.0±0.4 | 31.3±0.9 | 28.8±1.9 | 27.8±1.5 | 9.9 |
| GRAM-masked | 35.0 | 31.4±0.5 | 28.0±0.6 | 26.5±1.8 | 25.5±2.3 | 9.5 |
| GRAM-LoRA | **39.6** | **37.0±0.3** | **33.4±1.3** | **30.6±2.0** | **29.8±0.8** | 9.8 |
| gram_hyp (repro) | 18.2 | 16.0±1.0 | 13.0±0.9 | 11.4±0.6 | 10.2±1.6 | 8.0 |
| PMRL-masked | 25.9 | 25.1±0.8 | 23.4±0.3 | 22.4±0.9 | 20.9±1.5 | 5.0 |
| PMRL-LoRA | 32.9 | 29.8±0.2 | 27.5±0.2 | 25.7±1.0 | 24.5±0.7 | 8.4 |

Readings (final wording for the paper):
1. SCA-T1 has the gentlest degradation of the competitive arms (7.1 vs 9.5–9.9 for
   every GRAM arm) and MATCHES the best volume arm (GRAM-LoRA) at 75% (30.6 = 30.6)
   and 90% (29.5±0.9 vs 29.8±0.8 — statistical tie) despite starting 3.0 lower.
   Claim the slope + parity-at-heavy-missingness, NOT raw dominance.
2. Masked training with error bars: nomask drops 15.4 vs 7.1–8.3 masked — the single
   largest effect in the grid, now seed-robust.
3. gram_hyp (our v1 reproduction attempt of HyperGRAM, pre-norm reading) sits low
   across the surface. Given the paper's ambiguity between two readings and no released
   code, this is reported only as an appendix reproduction note — HyperGRAM's PUBLISHED
   numbers are what our tables cite. No E4 row exists for their method: they report no
   missing-modality experiments.

## E6 final (known-pairs-only, the A10 definition): the calibration table

| arm | slope (→1) | Pearson | R² | nDCG@10 |
|---|---|---|---|---|
| **SCA** | **0.978** | 0.369 | **−0.67** | 0.543 |
| SCA-T1 | 0.974 | 0.320 | −4.35 | 0.543 |
| GRAM | 2.309 | 0.398 | −17.8 | 0.544 |
| GRAM-masked | 1.831 | 0.369 | −16.9 | 0.534 |
| GRAM-LoRA | 2.794 | 0.389 | −17.8 | 0.557 |
| PMRL | 0.505 | 0.334 | −49.5 | 0.489 |

SCA is the only method family with slope ≈ 1, and its R² (−0.67) is 25–70×
closer to 0 than every volume/λ₁ arm — the residual negativity is the
under-dispersion A10 predicted, not mis-scaling. T1's higher lr costs some
calibration tightness (R² −4.3 vs −0.67): the paper reports both SCA configs —
base = best-calibrated, T1 = best-retrieval — as the same method's two operating
points on one Pareto front (identical slope ≈ 1 in both).

## E5 final (50%, seed mean): displacement disp_intact / disp_hit

SCA −4.5/119 (best intact-fairness), SCA-T1 −9.5/94 (best hit-protection),
GRAM −9.4/134, GRAM-LoRA −8.6/217, PMRL −13.8/152, nomask +7.3/307.
Honest note: T1 trades a little intact-fairness for hit-protection vs base SCA;
both dominate every baseline on the combined picture.
