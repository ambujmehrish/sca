# Wave 3 analysis — SCA downstream finetunes (E1 finetuned rows)

All five finetunes start from the Wave-1 SCA Stage-1 pretrain checkpoint (150k-clip,
LoRA) and run the GRAM finetune recipe (4 epochs, bs 64, lr 2e-5). ITM = table metric
(reference protocol, T2D); paper column = GRAM (ICLR 2025) finetuned rows.
CORRECTION (verified against GRAM Sec 4.1): GRAM's paper ALSO pretrains on the 150k
VAST-27M subset — the paper delta is an EVAL-ENVIRONMENT offset (their env scores the
same released checkpoint ~2 R@1 hotter than ours) plus finetune-recipe differences,
NOT pretrain scale.

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
2. **The 5.5–7-point gaps to the paper's ft rows are NOT pretrain scale** (corrected:
   GRAM's paper also pretrains on the 150k subset). Known contributors: the measured
   eval-environment offset (~2 R@1 on the same released checkpoint: 54.8 published vs
   52.5 in our env) and finetune-recipe differences; the remainder is unattributed
   until the GRAM-repro finetune (same recipe, same env, from `workdir_pretrain/gram`
   best) lands — THAT number is the only valid finetuned comparison.
3. **VATEX exceeds the paper by 6.5** (94.2 vs 87.7) — RESOLVED: not comparable. The
   HyperAlign VATEX test annotation is `descs_ret_test_431.json`, a 431-clip
   audio-complete subset, not the standard 1,500-video split; a smaller gallery inflates
   R@1. Our VATEX rows must be footnoted "431-clip audio-complete subset" and never
   bolded against the paper's number.
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
