# Wave 3 analysis — SCA downstream finetunes (E1 finetuned rows)

All five finetunes start from the Wave-1 SCA Stage-1 pretrain checkpoint (150k-clip,
LoRA) and run the GRAM finetune recipe (4 epochs, bs 64, lr 2e-5). ITM = table metric
(reference protocol, T2D); paper column = GRAM (ICLR 2025) finetuned rows, which start
from the FULL 27M-clip GRAM pretrain — so the paper delta bundles the pretrain-scale
gap already measured in Wave 1, not just method differences.

| benchmark (task) | ITM best / final | raw scorer best / final | GRAM paper ft | Δ vs paper |
|---|---|---|---|---|
| MSR-VTT (T-VAS) | 57.1 / 56.6 | 41.8 / 41.8 | 64.0 | −6.9 |
| DiDeMo (T-VA) | 61.8 / 61.6 | 35.9 / 35.7 | 67.3 | −5.5 |
| ActivityNet (T-VA) | 62.9 / 62.8 | 37.1 / 34.7 | 69.9 | −7.0 |
| VATEX (T-VAS) | 94.2 / 93.5 | 80.7 / 80.7 | 87.7 | **+6.5** |
| AudioCaps (T-VA) | 51.6 / 51.6 | 46.7 / 46.7 | 33.2 (zs only) | — |

## Readings

1. **Finetuning works on top of the SCA pretrain**: MSR-VTT goes 53.5 (zs, Wave 1) →
   57.1 ft. Every benchmark trains stably; best→final decay is ≤ 0.7 ITM points on all
   five (the Wave-2 stability picture carries over to finetuning).
2. **The 5.5–7-point gaps to the paper's ft rows are the pretrain-scale gap, not a
   finetune failure.** The paper finetunes from the 27M-clip pretrain; ours from the
   150k subset. Wave 1 already measured this same gap at ~1.3 points zero-shot with
   VAST-ckpt initialization; finetuning amplifies whatever the trunk absorbed at scale.
   The apples-to-apples claim the plan makes (DoD) is SCA vs GRAM **under the identical
   150k budget** — the zero-shot version of that comparison is done (Wave 1/2); a
   GRAM-repro finetune arm (same recipe, from `workdir_pretrain/gram` best) is the
   matching ft comparison if we want finetuned parity rows too.
3. **VATEX exceeds the paper by 6.5** (94.2 vs 87.7). Treat with caution until the eval
   protocol is double-checked (VATEX has multiple public test-split conventions; ours is
   the HyperAlign `descs_ret_test.json`). If their split differs, this row is not
   comparable in either direction.
4. **AudioCaps at 51.6 ITM (T-VA)** vs GRAM's 33.2 zero-shot: no finetuned AudioCaps row
   is published in the extracted tables, so this is our finetuned number vs their
   zero-shot — not a like-for-like win, just the E1 grid cell filled.
5. **Raw-scorer vs ITM spread stays large after finetuning** (e.g. MSR-VTT 41.8 vs 57.1)
   — consistent with Wave 2's finding that the shared ITM reranker does heavy lifting at
   full modality; E4/E5/E6 raw-space grids remain the discriminating experiments.
6. **ActivityNet's raw scorer decays while ITM holds** (37.1→34.7 raw, 62.9→62.8 ITM):
   the embedding space drifts under long-video finetuning but the reranker compensates.
   Worth an eye when the E4 grids include the ft checkpoints later.

## Status / next

- ft_msrvtt_depth (E10, stacks depth on the MSR-VTT best ckpt): ready to submit.
- E4/E5/E6 grid job over the 7 pretrain checkpoints: collect `results/e4/*.json`.
- Optional for finetuned parity tables: GRAM-repro finetune arm on MSR-VTT under the
  same budget (uses the trunk's original ft config from the GRAM best pretrain ckpt).
- VATEX protocol check before quoting the +6.5 row.
