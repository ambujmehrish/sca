# HyperGRAM (Na et al., CVPR 2026) — equal-budget competitor analysis

"Hyperbolic Gramian Volumes for Multimodal Alignment", UT Arlington. Hybrid
Euclidean/hyperbolic Gramian volume (Lorentz model, learnable mixing α→0.51, +1 scalar
param, keeps GRAM's DAM loss). Motivation: Euclidean volumes variance-collapse under L2
norm (std 0.005 around det≈1); hyperbolic inner products preserve variance (std 0.12).

## Why it matters to us

**Same experimental budget as ours — and as everyone's**: GRAM (paper), PMRL (paper)
and HyperGRAM all continue-pretrain from the VAST checkpoint on the 150k subset
(verified verbatim in all three papers). The 150k setting IS the family standard;
the 27M scale lives only in the shared VAST initialization. HyperGRAM is a direct
competitor row at the same budget.

## Their zero-shot R@1 (their Table 1) vs ours

| | MSR-VTT T2V/V2T | DiDeMo | ActivityNet | VATEX(std split) |
|---|---|---|---|---|
| GRAM (their eval, 150k) | 54.8 / 52.1 | 49.8 / 48.5 | 56.2 / 49.6 | 77.0 / 74.9 |
| Pure Hyperbolic | 54.8 / 52.5 | 49.1 / 48.3 | 57.0 / 50.9 | 76.7 / 74.3 |
| **HyperGRAM** | **56.6 / 53.6** | **51.3 / 49.5** | **58.2 / 51.8** | **79.9 / 75.7** |
| Δ over their GRAM | +1.8 / +1.5 | +1.5 / +1.0 | +2.0 / +2.2 | +2.9 / +0.8 |
| — | | | | |
| GRAM-repro (our eval) | 52.6 / 49.3 | 49.6 / 48.5 | 52.0 / 46.2 | (431 subset) |
| SCA (our eval) | 53.5 / 49.9 | 50.0 / 47.3 | 52.6 / 48.3 | (431 subset) |
| Δ over our GRAM | +0.9 / +0.6 | +0.4 / −1.2 | +0.6 / +2.1 | — |

## Protocol audit — absolutes are NOT cross-comparable

CORRECTED: GRAM's published numbers are themselves 150k-budget (their Sec 4.1:
"further pretrain ... on a small subset of VAST27M comprising 150k samples"), so
HyperGRAM's GRAM row (54.8) simply cites GRAM's published same-budget number —
internally consistent. The non-comparability is ENVIRONMENTAL, and we can measure it
exactly: the same officially released GRAM checkpoint scores 54.8 (their/GRAM's env),
53.4 (HyperAlign's env), 52.5 (ours). Published rows in this family run ~2 R@1 hotter
than our environment (undisclosed eval details — no reranking mention, no R@10, no
code). Cross-paper comparisons must therefore use each method's Δ over its own GRAM
baseline, or a single shared environment, not absolutes.

Honest reading of the deltas: their zero-shot gain over GRAM (+1.5…+2.9 T2V) is larger
than ours (+0.4…+0.9). Zero-shot full-modality R@1 is their strong axis.

## What they do NOT have (our unique surface, unchanged)

- **No missing-modality experiments** — and their method inherits volume's structural
  problems there: reduced-arity handling is unaddressed, and the mean-imputation
  degeneracy we proved (singular Gram ⇒ volume 0) applies verbatim to the hyperbolic
  Gram (an imputed row still lies in the span of present rows ⇒ det≈0 up to the
  Lorentz timelike component). E4 remains ours alone.
- No calibration (E6), no cardinality-bias analysis (E5), no stability trajectories,
  no raw-vs-reranked transparency, no cross-dataset raw-space transfer, no finetuning,
  no audio benchmarks (AudioCaps/VGGSound), no k=5 arity result.
- Zero-shot R@1 on four video benchmarks is their entire empirical surface.

## Consequences for our paper

1. Cite it; add its rows to the 150k tier of the tables (flagged: different eval env).
2. The zero-shot gap raises the stakes on the tuning wave (A7 gates / A8 / A6): our
   zs R@1 needs the lift for the full-modality columns.
3. Position: HyperGRAM makes the volume more discriminative at full modality;
   SCA makes the score usable under missingness (robust, calibrated, arity-fair,
   transferable). Orthogonal axes; theirs collapses exactly where our headline lives.
4. Their variance-collapse analysis (deg≈1, std 0.005) independently corroborates our
   E6 finding that volume scores are poorly scaled (slope 2.4–3.2) — quotable support.
