# HyperGRAM (Na et al., CVPR 2026) — equal-budget competitor analysis

"Hyperbolic Gramian Volumes for Multimodal Alignment", UT Arlington. Hybrid
Euclidean/hyperbolic Gramian volume (Lorentz model, learnable mixing α→0.51, +1 scalar
param, keeps GRAM's DAM loss). Motivation: Euclidean volumes variance-collapse under L2
norm (std 0.005 around det≈1); hyperbolic inner products preserve variance (std 0.12).

## Why it matters to us

**Same experimental budget as ours**: they pretrain on VAST150k for 1 epoch on the VAST
trunk (EVA-CLIP ViT-g/14 + BEATs + BERT-base) — the equal-budget protocol our campaign
uses. Independent confirmation that the 150k setting is the emerging standard for this
comparison (good for defending our setup), and a direct competitor row for our tables.

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

Their GRAM@150k MSR-VTT cell is 54.8 — exactly the GRAM paper's 27M-pretrain number,
and +2.3 above what the OFFICIAL 27M GRAM checkpoint scores through our
official-checkpoint-validated pipeline (52.5; HyperAlign reference eval of the same
ckpt: 53.4). A 1-epoch 150k rerun out-scoring the official 27M checkpoint under an
identical protocol is implausible; their eval protocol is systematically hotter by
~1.5–2.5 points (undisclosed eval details — like GRAM's paper, no mention of the
reranking stage; no R@10 anywhere; no code released to check). Cross-paper comparisons
must therefore use each method's Δ over its own GRAM baseline, not absolutes.

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
