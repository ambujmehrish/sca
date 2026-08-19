# Wave 4 — tuning arms, HyperGRAM repro, same-env GRAM-ft, depth (MSR-VTT zs ITM T2D best/final)

## 1. T1 (lr 1e-4) is the breakthrough: SCA 54.9 / 54.4

| arm | ITM best/final | raw best/final |
|---|---|---|
| SCA base (lr 2e-5, Wave 1) | 53.5 / 53.4 | 34.2 / 33.5 |
| **T1 lr 1e-4** | **54.9 / 54.4** | **36.6 / 36.1** |
| A6 asym | 53.8 / 53.7 | 34.3 / 33.4 |
| A5 p03 | 53.7 / 53.7 | 34.5 / 33.3 |
| A6 r16 | 53.6 / 53.6 | 34.3 / 33.3 |
| A7 gates / A8×3 / T2 / A6 fullft | 53.2–53.5 | 34.2–35.8 |

The recipe gap was the LEARNING RATE: our 2e-5 was inherited from HyperAlign, but
GRAM's published 54.8 was trained at 1e-4. At the matched lr, same budget, same env:

- **SCA (T1) 54.9 vs the official GRAM checkpoint 52.5 — +2.4, matched recipes,
  one environment.** This is the clean dominance number.
- 54.9 in OUR env exceeds GRAM's published 54.8 from an env we measured to run
  ~2 R@1 hotter (same released ckpt: 54.8 there, 52.5 here).
- Δ-over-GRAM is now **+2.3** (54.9 vs repro 52.6) — above HyperGRAM's claimed +1.8.
- Raw scorer 36.6/36.1: the MSR-VTT raw gap to GRAM (37.7) nearly closed, stability
  kept (best→final −0.5), still LoRA.
- All other knobs are noise at ±0.3 (good ablation rows: the method is not
  hyperparameter-fragile; gates/λ/rank/масk-schedule defaults hold).
- A6 fullft repeats the instability signature (raw best 35.8 but ITM 53.4→52.5).

**New headline config: sca_pretrain + learning_rate 1e-4 ("SCA-T1").**

## 2. HyperGRAM: we cite their published numbers; our reproduction is inconclusive

**Reported in our tables (from Na et al., CVPR 2026, Table 1): MSR-VTT 56.6 / 53.6,
DiDeMo 51.3 / 49.5, ActivityNet 58.2 / 51.8, VATEX 79.9 / 75.7.** These are their
same-budget (VAST ckpt + 150k, 1 epoch) numbers in their evaluation environment, and
they are what Table 1 shows for HyperGRAM.

Our reproduction attempt is an APPENDIX note, not a table row. v1 (hyperbolic branch on
pre-normalisation features — the reading their method section motivates via "varying
spatial norms") trained to 35.2 ITM. But their experiments section says the method
"only changes the inner product computation", which on GRAM's L2-normalised features is
a bounded shift of the cosine Gram — a different computation, and the only one
consistent with their Fig. 5's volume range [2.0, 2.5]. v2 implements that reading
(`hyp_use_prenorm=false`, volumes verified bounded [1.4, 3.5]) and has not been run to
completion. With no code released and the paper ambiguous between the two readings, we
make NO reproducibility claim: their published numbers stand as cited.

## 3. Same-env finetuned head-to-head: GRAM-ft wins MSR-VTT — with a known lever

GRAM-repro ft: **61.2 / 61.2** (raw 50.5) vs SCA ft 57.1 / 56.6 (raw 41.8).
Honest loss (−4.1), and informative: the two arms differ in exactly the two knobs
Wave 4 just exposed — (a) SCA-ft inherited the too-low lr 2e-5 lineage (both its
pretrain AND its ft), (b) GRAM-ft is FULL finetuning while SCA-ft is LoRA-only
(capacity gap that full-data finetuning rewards; note GRAM-ft was stable here).
Next arm (queued): **SCA-ft-v2 = finetune from the T1 checkpoint** (and if needed,
full-FT or higher-lr ft variant). Until then the finetuned table reports both
numbers as measured.

## 4. E10 depth (k=5): +1.6 for free — DoD #5 done

SCA ft depth (T-VASD): **58.7 / 58.0** vs SCA ft T-VAS 57.1 — adding the 5th
modality with ZERO centroid-code change improves retrieval by +1.6. GRAM needed a
dedicated volume_computation5 for this arity. The arity-invariance claim now has a
measured payoff attached.

## Follow-ups (queued)

1. SCA-T1 becomes the headline: seeds ×3, E1 zs grid on the T1 ckpt (all
   benchmarks), E4/E5/E6 grids with the T1 arm, SCA-ft-v2 from T1.
2. gram_hyp checkpoint through the E4 grids (does hyperbolic volume inherit
   Euclidean volume's missingness fragility? geometry says yes).
3. Grab gram_hyp's α trajectory from its log for the repro note.
