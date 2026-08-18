# Campaign status — results in hand vs paper needs (2026-08-18)

One page: what is measured, what each measurement supports, what still blocks the paper.
Sources: `wave1/`, `wave2/`, `wave3/`, `a10_flickr8k_k2.md`, `wave1/validation_official_gram.md`.

## 0. Foundations (done, citable in the paper)

- **Pipeline validated against the authors' checkpoint**: official GRAM ckpt through this
  code scores 52.5 ITM R@1 on MSR-VTT zs T-VAS vs 53.4 via the reference pipeline (Δ0.9 =
  environment noise). Protocol identical by construction (verbatim data_cfg inheritance).
- **Metric protocol established**: published GRAM-family table numbers are ITM-reranked;
  raw scorers (volume / centroid / λ₁) are embedding-space diagnostics. Both are logged
  for every run.
- **Training validated**: our GRAM-repro (52.6) ≡ official ckpt (52.5) under identical eval.
- **A10 decided the calibration mechanism** (k=2, 3 seeds, real data): KL + regression,
  fixed τ=0.07, known-pairs-only — slope 1.027≈1 (the property E5/E6 need), no retrieval
  cost. Shipped as the default config.
- GRAM path byte-for-byte regression-tested (present=None); 89+1 unit tests green.

## 1. Results in hand

### Zero-shot MSR-VTT T-VAS — all 8 pretrain arms, identical 150k budget (Waves 1–2)

| arm | adapter | train-mask | ITM best/final | raw best/final |
|---|---|---|---|---|
| **SCA** | LoRA | schedule | 53.5 / **53.4** | 34.2 / 33.5 |
| SCA-nomask | LoRA | off | 53.4 / 53.2 | 34.2 / 34.0 |
| GRAM | full-FT | off | 52.6 / 51.1 | 37.7 / 32.2 |
| GRAM-masked | full-FT | schedule | 52.1 / 48.3 | 35.0 / 24.9 |
| GRAM-LoRA | LoRA | off | 53.3 / 50.1 | 39.6 / 30.0 |
| PMRL | full-FT | off | 51.9 / 46.9 | 26.8 / 22.9 |
| PMRL-masked | full-FT | schedule | 52.6 / 45.9 | 25.9 / 20.5 |
| PMRL-LoRA | LoRA | off | 53.9 / 53.3 | 32.9 / 32.6 |
| official GRAM ckpt | — | — | 52.5 | 38.7 |
| GRAM paper (27M pretrain) | — | — | 54.8 | — |

Supported findings (wave2/ANALYSIS.md): SCA ≥ every GRAM arm at equal budget; stability
is the objective, not the adapter (GRAM-LoRA decays −3.2 with the identical LoRA budget);
masked training is free for SCA, costly for GRAM (−0.5/−2.8); masking keeps the audio
modality alive (A→T 5.3 vs 0.4 without).

### Finetuned SCA — Wave 3 (ITM best; paper column = GRAM ft from 27M pretrain)

| benchmark | SCA ft | GRAM paper ft |
|---|---|---|
| MSR-VTT T-VAS | 57.1 | 64.0 |
| DiDeMo T-VA | 61.8 | 67.3 |
| ActivityNet T-VA | 62.9 | 69.9 |
| VATEX T-VAS | 94.2 (†split check) | 87.7 |
| AudioCaps T-VA | 51.6 | 33.2 (zs only) |

Gaps vs paper = pretrain-scale (150k vs 27M), consistent with Wave 1. Same-budget ft
comparison (GRAM-repro ft) not run — optional arm.

## 2. Experiment grid: status

| id | what | status |
|---|---|---|
| E1 zs retrieval | MSR-VTT ✅ (8 arms). DiDeMo/ANet/VATEX zs of our ckpts | ❌ **not run — needs zs eval configs per benchmark (cheap eval jobs)** |
| E2 ft retrieval | SCA ✅ 5 benchmarks (wave3). GRAM rows imported ✅ | partial by design |
| E3 AudioCaps T2A ✅ ft (51.6) · VGGSound-5K zs classification | ❌ not run |
| **E4 (headline)** missingness 2×2 | launcher fixed+versioned (`slurm_scripts/e4_grid.sh`), **blocked on SLURM controller — top priority** |
| E5 cardinality calibration | produced by the same e4_grid job |
| **E6 (headline)** S-vs-S* | same job (test-caption S* cache auto-built) |
| E7 L_concept vs missing rate | ❌ needs A5/A6-style arms — cut candidate |
| E8 diagnostics | runs on `results/e4/feats/*.pt` once E4 dumps exist (CPU) |
| E9 efficiency | ⚠ compile from existing logs/sacct — no new GPU time needed |
| E10 depth k=5 | `ft_msrvtt_depth_sca.sh` ready — submit when SLURM returns |
| A10 calibration | ✅ done (decision shipped) |
| A1–A9 ablations | configs generated (26 arms); **none run**; priority A1/A3/A5; A2+E8 are CPU on E4 feats |
| ×3 seeds (DoD 6) | ❌ single seed everywhere so far |

## 3. DoD scorecard

1. E1–E3 tables with imported rows — **partial** (MSR-VTT zs full; ft SCA rows done; other zs benchmarks missing).
2. E4 graceful degradation + E5 near-zero cardinality bias — **blocked on the e4_grid job** (the paper's central claim).
3. E6 R² under A10 config — same job.
4. A2 first-order sufficiency — CPU on E4 feats, after E4.
5. E10 zero-code-change k=5 — ready to submit.
6. ×3 seeds + per-row configs + CI regression — regression test ✅, configs ✅, seeds ❌.

## 4. Critical path when SLURM returns (in order)

```
sbatch slurm_scripts/e4_grid.sh                 # E4+E5+E6 (headline), 1 GPU, ~6h
sbatch slurm_scripts/ft_msrvtt_depth_sca.sh     # E10, stacks on ft_msrvtt best
# seeds for the headline zs trio (DoD 6):
sbatch slurm_scripts/run_config.sh config/sca/pretrain_cfg/sca_pretrain.json workdir_pretrain/sca_s51 --seed 51
sbatch slurm_scripts/run_config.sh config/sca/pretrain_cfg/sca_pretrain.json workdir_pretrain/sca_s52 --seed 52
sbatch slurm_scripts/run_config.sh config/baselines/pretrain_cfg/gram_pretrain.json workdir_pretrain/gram_s51 --seed 51
sbatch slurm_scripts/run_config.sh config/baselines/pretrain_cfg/gram_pretrain.json workdir_pretrain/gram_s52 --seed 52
```
Then: A1/A3/A5 ablation arms (MANIFEST), E8+A2 on the E4 feature dumps (CPU), LaTeX
tables via `benchmark_eval/make_latex_tables.py`.

## 5. Writable today (not blocked on anything)

Method + guards; A10 mechanism section; equal-budget zs table + stability/masking
analysis; SCA ft rows with the scale-gap framing; pipeline-validation appendix;
E9 from existing logs. The paper's empirical core that remains outstanding is exactly
one GPU job (E4/E5/E6) plus seeds.
