# Wave 2 analysis — the 2×2 masking arms + LoRA parity (MSR-VTT zs T-VAS)

All eight pretrain arms, identical data/recipe/eval. ITM = table metric (reference
protocol); raw = each method's own embedding scorer (the space E4/E5/E6 probe).

| arm | adapter | train-mask | ITM best (step) | ITM final | raw best / final |
|---|---|---|---|---|---|
| **SCA** | LoRA | schedule | 53.5 (2119) | **53.4** | 34.2 / 33.5 |
| SCA-nomask | LoRA | off | 53.4 (2119) | 53.2 | 34.2 / 34.0 |
| GRAM | full-FT | off | 52.6 (529) | 51.1 | **37.7** / 32.2 |
| GRAM-masked | full-FT | schedule | 52.1 (529) | 48.3 | 35.0 / 24.9 |
| GRAM-LoRA | LoRA | off | 53.3 (794) | 50.1 | **39.6** / 30.0 |
| PMRL | full-FT | off | 51.9 (264) | 46.9 | 26.8 / 22.9 |
| PMRL-masked | full-FT | schedule | 52.6 (264) | 45.9 | 25.9 / 20.5 |
| PMRL-LoRA | LoRA | off | **53.9** (1324) | 53.3 | 32.9 / 32.6 |
| official GRAM ckpt (reference) | — | — | 52.5 | — | 38.7 |

## Readings

1. **Peaks cluster within ~2 points (51.9–53.9) on the ITM metric** — the shared VAST
   cross-encoder reranker compresses method differences at full modality. The method
   separation lives in the raw embedding space, exactly where the E4/E5/E6 grids operate.
2. **Stability is NOT explained by the adapter alone.** SCA (−0.1 best→final), SCA-nomask
   (−0.2) and PMRL-LoRA (−0.6) hold; GRAM-LoRA decays −3.2 *despite the identical LoRA
   budget* (and its raw score collapses 39.6→30.0). With adapter parity, GRAM's volume
   objective still drifts — SCA's objective is doing real work, not just the adapter.
   Every full-FT arm decays 1.5–6.7 points.
3. **Masked training is free for SCA, costly for GRAM.** SCA vs SCA-nomask at full
   modality: 53.5/53.4 vs 53.4/53.2 (indistinguishable — the E4 prerequisite "masked
   training costs nothing when nothing is missing" holds). GRAM-masked loses vs GRAM
   (52.1/48.3 vs 52.6/51.1): reduced-arity volume training destabilises the volume.
4. **Masking protects the weak modality.** SCA-nomask's audio alignment collapsed late
   (cosine A→T final 0.4 R@1); SCA's masked schedule kept it alive (5.3). Direct evidence
   for the masking mechanism, before E4 even runs.
5. **PMRL-LoRA's 53.9 peak rides the reranker** (raw scorer only 32.9): the λ₁ embedding
   space stays weak; the ITM stage does the lifting. E4 (raw-space grids) is expected to
   expose this.

## Status / next

- E4/E5/E6 grid job over 7 checkpoints: submitted (results/e4/*.json when done).
- Wave 3 finetunes: msrvtt running, others queued; depth (E10) after msrvtt.
- Headline rows still need x3 seeds (plan DoD): rerun `sca`, `gram`, `gram_lora` with
  `--seed 51` / `--seed 52` once the grid confirms the story.
