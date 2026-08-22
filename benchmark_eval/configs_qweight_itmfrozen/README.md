# Stage 2 on frozen weights (`itm_lora_off`)

Identical to `configs_qweight/` except for one key, `itm_lora_off: true`. The dual encoder
still runs with the trained adapters; the ITM reranker -- its cross-encoder BERT and the
`condition_feats` fed into it -- runs through the frozen backbone.

## Why

The reported metric is produced by two stages that were trained differently.

Stage 1, the contrastive scorer, is what LoRA was trained for. Stage 2 is a cross-encoder
plus an ITM head that arrived pretrained in the VAST foundation checkpoint and was never
trained again here. But the retrieval loss reaches that cross-encoder's BERT through the
same `multimodal_encoder` adapters, and its `condition_feats` come from the adapted vision
and audio encoders. So every LoRA step moves stage 2 away from the calibration its own frozen
head was fitted to, driven by an objective that is not ITM.

The measured results have that shape, and they have it on **every** benchmark. T9
(`workdir/e1_frames`) against GRAM's released checkpoint (`workdir/e1_zs`), same environment,
same eval task per benchmark:

| benchmark | SCA aggr | GRAM aggr | lead | SCA after ITM | GRAM after ITM | lead |
|---|---|---|---|---|---|---|
| MSR-VTT     | 45.2 | 38.7 | **+6.5** | 54.8 | 52.5 | +2.3 |
| DiDeMo      | 34.2 | 28.2 | **+6.0** | 51.5 | 50.7 | +0.8 |
| ActivityNet | 34.4 | 31.0 | **+3.4** | 55.8 | 56.3 | **−0.5** |
| VATEX       | 81.7 | 75.6 | **+6.1** | 90.5 | 90.0 | +0.5 |
| AudioCaps   | 27.1 | 22.9 | **+4.2** | 35.2 | 32.2 | +3.0 |

SCA hands the reranker a better candidate set on all five, by +3.4 to +6.5. After reranking
the lead is +2.3, +0.8, −0.5, +0.5, +3.0. The reranking stage removes 1.2 to 5.6 points of
lead on every benchmark, and on ActivityNet it removes more than there was. That is not a
per-benchmark quirk; it is one stage behaving the same way five times.

## What this measures, and what it does not

These configs evaluate checkpoints that were **trained** with the adapters in the ITM branch.
Running them without it is therefore a train/test mismatch by construction: it is a
diagnostic, not a recipe. It answers one question -- is stage 2 degraded by adapter drift? --
and nothing else.

If it wins, the recipe follow-up is to retrain with `itm_lora_off` set, so the ITM branch is
fitted on the same weights it is scored with. `_itm_lora_ctx()` is applied in the training
loss too, so that arm needs only the flag in the pretrain config.

If it loses, the adapter drift is not the cost, and the reranking gap has another cause.

## Cost

The flag adds a second vision/audio encoder pass per batch (the frozen `condition_feats`),
so eval takes roughly twice as long per cell. Nothing is retrained.
