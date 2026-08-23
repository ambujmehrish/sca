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

## Our reimplementation vs theirs — line-by-line

Read after cloning their repo. Every row is a substantive difference, not a style choice, and
several change the objective materially. `model/baselines.py::GRAMHyp` and
`utils/volume.py::volume_computation_lorentz` are ours; `utils/hyperbolic_volume.py` and
`model/gram.py` are theirs.

| | ours (`gram_hyp`) | theirs |
|---|---|---|
| hyperbolic map | `pi(x) = [sqrt(1+\|\|x\|\|^2), x]` inline, fixed unit curvature | `exp_map0(v, curv)` onto the hyperboloid, then `lorentz_inner` |
| curvature | **none** — hard-wired to 1 | `nn.Parameter`, `curvature_init 1.0`, `learn_curvature true`, optionally per modality |
| curvature lr | n/a | **its own optimizer group at 10x the base lr** (`build_optimizer.py:60`) |
| scale matching | **none** — the two volumes are mixed raw | hyperbolic volume rescaled to the Euclidean mean before mixing: `scale = euc.mean()/hyp.mean()`, detached, clamped [0.001, 10] |
| mixing weights | one `alpha`, used as `alpha*V_euc + (1-alpha)*V_hyp`, clamped [0,1] | two independent parameters `euclidean_weight`, `hyperbolic_weight`, not tied to sum to 1 |
| feature source | `hyp_use_prenorm` switches between pre-normalisation projections and normalised features | one feature set, no such switch |

The scale-matching row is the one most likely to dominate. Mixing a Euclidean and a Lorentzian
volume without normalising their magnitudes lets whichever branch is numerically larger decide
the loss, and the learnable weight then has to fight the scale gap rather than express a
preference. Their code removes that before the weights ever apply.

The last row is worth stating plainly: `hyp_use_prenorm` was introduced to resolve an
ambiguity we believed existed in their paper. Their code has no such switch. Both v1 and v2
were answering a question that was never open, and the "two defensible readings" framing --
repeated in commit messages, launcher comments and analysis notes -- should be retracted
wherever it appears.

## Pretrained encoders their config loads

From `configs/default_model_cfg.json` (`vision_encoder_type: evaclip01_giant`,
`audio_encoder_type: beats`, `vision_resolution: 224`) resolved through
`model/general_module.py` and `model/gram.py`. These are the SAME three encoders our runs
use — the fork is shared, so no row in the table differs in its backbone.

| role | encoder | weight file (relative to their root) | dim |
|---|---|---|---|
| vision | **EVA01-CLIP-g-14** (`evaclip01_giant`), 224px, patch 14 | `pretrained_weights/clip/EVA01_CLIP_g_14_psz14_s11B.pt` | 1408 |
| audio | **BEATs** iter3+ , AudioSet-2M | `pretrained_weights/beats/BEATs_iter3_plus_AS2M.pt` | 768 |
| text + multimodal fusion | **bert-base-uncased** (`BertForMaskedLM`, cross-attention layers are the ITM reranker) | `pretrained_weights/bert/bert-base-uncased` | 768 |

The whole stack is then initialised from the **VAST foundation checkpoint**
(`model_step_204994.pt`, `run_cfg.pretrain_dir`), so the encoders above are the architecture,
not the starting point — VAST's weights overwrite them wherever they overlap.

Subtitles use no separate encoder: `max_subtitle_len: 70` tokens go through the same BERT.

## Three directories their release omits

Not method differences — the published tarball just does not carry them. Each is supplied to
their tree without editing a line of their code:

| missing | what it is | how it is supplied |
|---|---|---|
| `evaluation_tools/` | vendored caption-eval package (pycocoevalcap et al.), imported at module level by `evaluation/evaluation_mm.py` | symlink to ours + `PYTHONPATH` |
| `pretrained_weights/` | the three encoder checkpoints above, loaded by RELATIVE path from inside model construction | symlink to ours, checked before launch |
| `datasets/` | the val annotation JSONs their config names (`datasets/annotations/<bench>/descs_ret_test.json`) | rewritten to ours by `make_hypergram_config.py` |

The encoder weights are verified in `hypergram_authors.sh` **before** srun, because their code
loads them by relative path deep inside model construction: a missing file otherwise appears
as a traceback on all four ranks after the data pipeline has spun up. `srun --chdir` puts the
ranks in their root so those relative paths resolve at all.

## Their repository also implements PMRL

`model/gram.py:93` -- `geometry_mode` accepts `'pmrl'`, `'pmrl_volume'` and `'hybrid_pmrl'`
alongside `'euclidean'`, `'hyperbolic'` and `'hybrid'`. So a PMRL row can be produced from
their code as well, and our `model/baselines.py` PMRL should be checked against it before the
reproduction row is trusted -- the same way this one was.
