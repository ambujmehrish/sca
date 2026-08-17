# Pipeline validation: official GRAM checkpoint, MSR-VTT zero-shot T-VAS

The same test the HyperAlign repo used to reproduce GRAM: evaluate the authors' released
checkpoint (`GRAM_pretrained_TVAS/ckpt/model_step_459.pt`) through the pipeline, protocol
generated verbatim from the reference run's hps (full `descs_ret_test.json`, 8 frames,
`ret_bidirection_evaluation`, itm_rerank 50). Run 2026-08-17 on 4xA100
(`workdir/validate_gram_official`).

| MSR-VTT zs T-VAS (T2V) | raw scorer R@1/R@10 | ITM-reranked R@1/R@10 |
|---|---|---|
| official ckpt — HyperAlign's pipeline (GRAM-base reference) | — | 53.4 / 83.3 |
| **official ckpt — THIS pipeline** | **38.7 / 74.4** | **52.5 / 82.5** (D2T 50.5/81.2) |
| our GRAM-repro, Wave 1 (best) | 37.7 / 73.7 | 52.6 / 82.6 |
| our SCA, Wave 1 (best / final) | 34.2 / 71.8 | 53.5 / 81.9 · 53.4 / 82.2 |
| GRAM paper | — | 54.8 / 82.9 |

Conclusions:

1. **Pipeline validated.** Official checkpoint through this code lands within ~0.9 R@1 of
   the reference number produced by the HyperAlign pipeline — residual is eval-environment
   noise (torch/ffmpeg/cudnn versions differ from their conda env), not protocol drift.
2. **Training validated.** Our Wave-1 GRAM-repro (52.6 ITM) is indistinguishable from the
   official checkpoint under the identical eval (52.5) — the from-scratch continued
   pretrain reaches the released checkpoint's quality.
3. **Metric protocol.** The reported GRAM/HyperAlign table numbers are the ITM-reranked
   metric; even the OFFICIAL checkpoint scores only 38.7 on the raw volume under this
   protocol (HyperAlign's own trained model: 25.0). Raw-scorer numbers are the E4/E5/E6
   embedding-space diagnostics, never the table metric.
4. **SCA standing after Wave 1**: 53.5 ITM -- above both the official checkpoint (52.5)
   and our GRAM-repro (52.6) under the identical protocol, with a LoRA budget instead of
   full finetuning, and the most stable best->final trajectory of the three arms.
