# HyperGRAM reproduction — status

**Their code IS released: https://github.com/uta-smile/HyperGram** (cloned read-only at
`/home/user/uta-smile/hypergram`). Everything written before 2026-08-23 assumed it was not,
including the claim that the method "admits two readings" — that ambiguity does not exist any
more, and no argument should be built on it.

## Their actual recipe (`configs/pretrain/pretrain_hybrid_vast150k_vatex_val_paper.json`)

| key | their value |
|---|---|
| `learning_rate` | **5e-05** |
| `train_epoch` / `epoch` | **1** |
| `batch_size` | 128 |
| `task` | **`ret%tvas%tv%ta`** (includes the subtitle task) |
| `vision_sample_num` | 2 |
| `audio_sample_num` | 1 |
| `valid_freq` | 20 |
| `first_eval` / `save_best` | true / true |
| `grad_norm` | 1.0 |
| `fp16` | true |
| `pretrain_dir` | `pretrained_models/pretrain_vast` |

Model side: `geometry_mode: hybrid`, `curvature_init: 1.0`, `learn_curvature: true`,
`initial_euclidean_weight: 0.5`, `initial_hyperbolic_weight: 0.5`,
`learn_hybrid_weights: true`, `gradient_clip_hyperbolic: 1.0`, `max_subtitle_len: 70`.

One detail that could not have been guessed: `utils/build_optimizer.py:60` puts every
parameter whose name contains `curvature` in its own group at **10x the base learning rate**
(`curvature_lr = model_cfg.get('curvature_lr', learning_rate * 10)`).

Their geometry is `utils/hyperbolic_volume.py` — Lorentz model, `exp_map0`, `lorentz_inner`,
and `hybrid_volume{2,3,4,5}` mixing a Euclidean and a hyperbolic Gramian volume by the
learnable weights above.

## Every arm we ran was at the wrong recipe

| arm | lr | epochs | task | verdict |
|---|---|---|---|---|
| `gram_hyp` (v1) | 2e-5 | 5 | `ret%tv%ta` | wrong on all three |
| `gram_hyp2` (v2) | 2e-5 | 5 | `ret%tv%ta` | wrong on all three |
| `H1_hypergram_paper` | 1e-4 | 5 | `ret%tv%ta` | wrong on all three — **cancelled** |

H1 was built from GRAM's recipe on the assumption that HyperGRAM followed it. It does not:
5e-5 rather than 1e-4, one epoch rather than five (5859 steps against ~1170), and it trains
the subtitle task ours omits. **No number from any of these three arms is evidence about
HyperGRAM**, and the 37.4 in particular must never be cited.

## What to do instead

Do not reimplement. Run **their** code with **their** config on our data — a real
reproduction rather than our reading of their paper. Their repository is the same VAST/GRAM
codebase we forked (`configs/`, `model/`, `evaluation/`, `run.py`, `utils/`), takes the same
VAST foundation checkpoint, and trains on the same 150k subset, so it should be close to
drop-in.

Whatever it produces is then labelled "HyperGRAM (authors' code, our environment)" — a far
stronger row than any reimplementation, and the only version that can be defended if a
reviewer knows the paper.

## Record of the error

The figure 37.4 was cited repeatedly as "our HyperGRAM reproduction does not work" — in commit
messages, in launcher comments, and in advice about the paper. It came from a run at the wrong
learning rate, the wrong epoch count and the wrong task mix, using our own reimplementation of
a method whose code was public the whole time. Anything written before 2026-08-23 citing 37.4,
51.0 or 35.2 as HyperGRAM's reproduced performance is wrong for that reason.
