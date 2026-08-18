# Paper tables — everything buildable from current results (2026-08-18)

Source of truth for the draft. Each table has a .tex twin in this directory.
Footnote symbols: † our VATEX = 431-clip audio-complete subset; * standard VATEX
split (not comparable with †); ‡ method uses extra tricks per GRAM's asterisk
(dual-softmax / higher res). Blocks A–B use the ITM-reranked protocol; "single-stage"
baselines have no reranker.

## T1 — Zero-shot T2V (main table)  [main_zeroshot_t2v.tex]

| Method | MSR-VTT T-VAS | DiDeMo T-VA | ANet T-VA | VATEX† T-VAS | AudioCaps T-VA |
|---|---|---|---|---|---|
| **A: equal budget (150k), measured** |
| **SCA (ours)** | **53.5**/81.9 | **50.0**/**78.4** | **52.6**/**85.7** | **90.3**/99.3 | **35.2**/**75.6** |
| SCA-nomask | 53.4/80.1 | — | — | — | — |
| GRAM (repro) | 52.6/82.6 | 49.6/76.6 | 52.0/84.1 | 88.9/**99.5** | 33.1/73.0 |
| GRAM-masked | 52.1/80.0 | — | — | — | — |
| GRAM-LoRA | 53.3/**83.1** | — | — | — | — |
| PMRL (repro) | 51.9/80.6 | — | — | — | — |
| PMRL-masked | 52.6/80.9 | — | — | — | — |
| PMRL-LoRA | 53.9/82.0 | — | — | — | — |
| *GRAM official ckpt (validation)* | *52.5/82.5* | — | — | — | — |
| **B: full-scale (~27M), reported** |
| GRAM (paper) | 54.8/82.9 | 54.2/79.3 | 59.0/91.1 | 83.5*/98.8 | 33.2/75.3 |
| PMRL (paper) | 54.5/— | 50.6/— | 56.0/— | 80.5*/— | 36.1/75.9 |
| VAST | 50.7/74.4 | 49.5/76.9 | 51.4/83.6 | 82.1*/96.8 | 32.1/65.4 |
| **C: single-stage (T-V), reported** |
| VideoPrism-b | 51.4/— | — | 49.6/— | 62.5*/— | — |
| mPLUG-2 | 47.1/79.0 | 45.7/71.1 | — | — | — |
| LanguageBind | 44.8/78.7 | 39.9/74.6 | 41.0/80.0 | — | 19.7/67.6 |
| UMT-L | 40.7/71.8 | 48.6/79.0 | 41.9/— | — | — |
| ImageBind | 36.8/70.0 | — | — | — | 9.3/42.3 |

## T2 — Zero-shot V2T  [main_zeroshot_v2t.tex]

| Method | MSR-VTT T-VAS | DiDeMo T-VA | ANet T-VA | VATEX† T-VAS | AudioCaps T-VA |
|---|---|---|---|---|---|
| **A: equal budget, measured** |
| **SCA (ours)** | 49.9/78.1 | 47.3/76.6 | **48.3**/**83.4** | **86.1**/**98.6** | **34.4**/**76.8** |
| GRAM (repro) | 49.3/80.9 | **48.5**/**76.8** | 46.2/79.4 | **86.1**/98.4 | 32.8/74.9 |
| GRAM-LoRA | **50.6**/80.9 | — | — | — | — |
| PMRL-LoRA | 49.8/79.8 | — | — | — | — |
| others (MSR-VTT only) | 49.3–50.1 | | | | |
| **B: full-scale, reported** |
| GRAM (paper) | 52.9/82.9 | 52.2/78.9 | 50.4/85.8 | 82.7*/98.1 | — |
| PMRL (paper) | 52.4/— | 48.4/— | 49.6/— | 75.2*/— | — |
| VAST | 49.0/76.2 | 48.2/78.6 | 46.8/77.4 | 78.7*/97.7 | — |

Honest note: v2t equal-budget is 3 wins, 1 tie (VATEX R@1), 1 loss (DiDeMo −1.2);
MSR-VTT v2t best in block A is GRAM-LoRA (50.6).

## T3 — Finetuned retrieval  [main_finetune.tex]

| Method | MSR-VTT T-VAS | DiDeMo T-VA | ANet T-VA | VATEX† T-VAS | AudioCaps T-VA |
|---|---|---|---|---|---|
| **A: 150k pretrain + GRAM ft recipe, measured (T2V / V2T R@1)** |
| SCA (ours) | 57.1 / 58.9 | 61.8 / 60.0 | 62.9 / 58.4 | 94.2 / 91.0 | 51.6 / 50.6 |
| GRAM (repro) ft | (running) | | | | |
| **B: full-scale, reported (T2V R@1/R@10)** |
| GRAM (paper) | 64.0/89.3 | 67.3/90.1 | 69.9/96.1 | 87.7*/100.0 | — |
| PMRL (paper) | 61.2/— | 70.2/— | 68.2/— | 84.1*/— | — |
| VAST | 56.6/79.4 | 65.6/88.1 | 68.8/95.5 | 87.5*/99.5 | — |
| **C: transcribed ft baselines (T2V)** |
| UMT-L‡ | 58.8/87.1 | 70.4/93.5 | 66.8/94.9 | 72.0*/— | — |
| vid-TLDR‡ | 58.5/86.9 | 70.4/94.0 | 65.2/94.5 | — | — |
| VALOR-L | 54.4/— | 57.6/— | 63.4/— | 76.9*/— | — |
| mPLUG-2 | 53.1/84.7 | 56.4/85.2 | — | — | — |
| T-MASS | 52.7/85.6 | 53.3/87.7 | — | 65.6*/97.2 | — |
| InternVideo-L‡ | 55.2/— | 57.9/— | 62.2/— | 71.1*/— | — |
| CLIP4Clip | 45.6/81.6 | 43.0/80.6 | 40.3/— | 63.0*/— | — |

## T4 — E4 missingness (raw space, MSR-VTT test, native scorers)  [e4_missingness.tex]

| arm (ckpt + scorer) | 0% | 25% | 50% | 75% | drop 0→75 |
|---|---|---|---|---|---|
| **SCA masked + centroid** | 34.2 | 32.2 | 27.7 | **29.4** | **4.8** |
| SCA nomask + centroid | 34.2 | 23.4 | 20.7 | 21.2 | 13.0 |
| GRAM + volume-(i) | **37.7** | **34.4** | 31.4 | 27.9 | 9.8 |
| GRAM-masked + volume-(i) | 35.0 | 31.1 | 27.8 | 26.0 | 9.0 |
| GRAM-LoRA + volume-(i) | 39.6 | 36.7 | **32.4** | 29.8 | 9.8 |
| GRAM + volume-(ii) imputed | 37.7 | 0.0 | 0.0 | 0.0 | collapse |
| PMRL-masked + λ₁ | 25.9 | 25.8 | 23.7 | 21.5 | 4.4 |
| PMRL + λ₁ | 26.8 | 24.0 | 21.5 | 21.2 | 5.6 |
| PMRL-LoRA + λ₁ | 32.9 | 29.8 | 27.6 | 25.2 | 7.7 |

(single seed; ±error bars and 90% column from the seeds rerun. Variant-(ii)
collapse is exact: mean-imputed row ⇒ singular Gram ⇒ volume 0 — proposition.)

## T5 — E5 cardinality bias (50% missing)  [e5_cardinality.tex]

| arm | score shift Δ(|M|=2→3) | intact-item rank bias | hit displacement | R@1@75%: raw → affine-calibrated |
|---|---|---|---|---|
| **SCA masked** | 0.068 | **−4.7** | **126** | 29.4 → 27.6 (no gain: pre-calibrated) |
| SCA nomask | 0.088 | +5.8 | 313 | 21.2 → 23.2 |
| GRAM | 0.050 | −8.8 | 146 | 27.9 → 28.9 (< SCA raw) |
| GRAM-masked | 0.057 | −11.3 | 137 | 26.0 → 28.6 (< SCA raw) |
| PMRL-masked | 0.148 | −11.6 | 162 | — |

## T6 — E6 semantic calibration @0% (PROVISIONAL: all-pairs fit; known-only rerun pending)  [e6_calibration.tex]

| arm | slope (→1 = calibrated) | Pearson | graded nDCG@10 |
|---|---|---|---|
| **SCA masked** | **1.010** | 0.261 | 0.543 |
| SCA nomask | 0.850 | 0.197 | 0.519 |
| GRAM | 2.822 | 0.280 | 0.544 |
| GRAM-masked | 2.372 | 0.281 | 0.534 |
| GRAM-LoRA | 3.211 | 0.272 | 0.557 |
| PMRL | 0.589 | 0.245 | 0.489 |
| PMRL-masked | 0.627 | 0.248 | 0.491 |

## T7 — Raw-space transfer (embedding scorers, no reranker, R@1)  [raw_transfer.tex]

| | MSR-VTT (selection bench) | DiDeMo | ANet | AudioCaps | VATEX† |
|---|---|---|---|---|---|
| SCA centroid | 34.2 | **32.7** | **33.3** | **26.1** | 71.9 |
| GRAM volume | **37.7** | 26.4 | 29.2 | 21.9 | **73.8** |
| Δ | −3.5 | **+6.3** | **+4.1** | **+4.2** | −1.9 |

(both checkpoints model-selected on MSR-VTT raw scores; no published counterpart
exists — GRAM's paper reports only reranked numbers.)

## T8 — Training stability (MSR-VTT zs ITM, best → final)  [stability.tex]

| arm | best | final | decay |
|---|---|---|---|
| **SCA** | 53.5 | 53.4 | **−0.1** |
| SCA-nomask | 53.4 | 53.2 | −0.2 |
| PMRL-LoRA | 53.9 | 53.3 | −0.6 |
| GRAM | 52.6 | 51.1 | −1.5 |
| GRAM-LoRA | 53.3 | 50.1 | −3.2 |
| GRAM-masked | 52.1 | 48.3 | −3.8 |
| PMRL | 51.9 | 46.9 | −5.0 |
| PMRL-masked | 52.6 | 45.9 | −6.7 |

## T9 — A10 calibration-mechanism ablation (k=2, Flickr8k, 3 seeds)  [a10_mechanism.tex]

| mechanism | slope | Pearson | nDCG@10 | R@1 | τ end |
|---|---|---|---|---|---|
| KL only (learnable τ) | 0.917 | 0.459 | 0.656 | 53.15±0.01 | 0.0777 (drifted +11%) |
| **KL + regression (fixed τ)** | **1.027** | **0.471** | 0.657 | **53.48±0.15** | 0.0700 |
| fixed τ = τ* | 0.350 | 0.460 | **0.705** | 43.66±0.07 | 0.5000 |

## T10 — Masked-training mechanism (SCA vs SCA-nomask)  [masking_mechanism.tex]

| property | SCA (masked) | SCA-nomask |
|---|---|---|
| full-modality ITM (best/final) | 53.5 / 53.4 | 53.4 / 53.2 |
| E4 drop 0→75% (raw) | **4.8** | 13.0 |
| audio cosine A→T final R@1 | **5.3** | 0.4 |
| hit displacement @50% | **126** | 313 |
| intact-item rank bias @50% | **−4.7** | +5.8 |

(masking costs nothing when nothing is missing — rows 1 — and buys rows 2–5.)

## Pending tables (data not yet available)

- GRAM (repro) ft row for T3 — job running.
- E4 error bars + 90% rate; E6 known-only — CPU rerun on cached feats.
- E10 depth (k=5) — job running. E9 efficiency — compile from sacct/logs.
- Tuning-wave arms (A5-A8) + remaining ablations; ×3 seed headline rows.
