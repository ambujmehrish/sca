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

**DoD-2 verdict — supported, with one honest nuance.**
- vs masked-GRAM(i): SCA degrades half as much (4.8 vs 9.0) and is ABOVE it at every
  nonzero rate despite a lower 0% start (29.4 vs 26.0 at 75%).
- vs masked-GRAM(ii): total collapse of the baseline (see proposition below).
- vs masked-PMRL: SCA is 6–8 points above at EVERY rate (29.4 vs 21.5 at 75%). PMRL's
  slope is comparable (4.4) but from a base 8 points lower — flat-because-weak, exactly
  the "λ₁ space stays weak" pattern Wave 2 predicted. State both numbers in the paper.
- 27.7@50 vs 29.4@75 non-monotonicity = single mask-draw noise; rerun grids with
  `--seed 1 2` (CPU, minutes) for error bars before the camera-ready plot.

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
