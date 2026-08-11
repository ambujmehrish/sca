# SCA — Spherical Centroid Alignment

SCA implemented alongside GRAM on the HyperAlign trunk (fork of the official GRAM/VAST
codebase + GatedHGNN additions). The GRAM path remains byte-for-byte intact and is
regression-tested (`present=None` == plain GRAM). Companion documents:
`SCA_Implementation_Plan_FINAL.md` (this repo's build plan) and
`SCA_Final_Research_Plan.pdf` (§4–7, theory/experiments).

## What SCA is

A gallery clip is represented by the **masked spherical mean** of its present modality
embeddings {V, A, S, (D)} on the unit sphere; text stays the query side only (leak-free,
same principle as the hypergraph arm). Scores are cosines in [-1, 1] at **every**
cardinality — the calibration GRAM's Gramian volume does not have — and the centroid is
arity-invariant by construction (k=2..5 with zero code change; E10).

Training objective: `L_align + α·L_sem + β·L_mask + δ·L_concept + λ·L_unif` (+ the trunk's
ITM), with an m†-masking schedule (`p_full` 1.0 → 0.5) whose masked view `mu_M` and full view
`mu_K` come from one forward pass.

## Layout (new vs. inherited)

| Path | Status |
|---|---|
| `model/gram.py`, `model/hypergraph.py` (GatedHGNN), encoders, `utils/*`, `evaluation/evaluation_mm.py`, `benchmark_eval/`, `slurm_scripts/` | inherited from HyperAlign, GRAM path unchanged |
| `model/sca.py` | SCA sibling of GRAM (`model_type: gram\|sca` config switch); swaps only the `forward_ret` loss block |
| `model/centroid.py` | masked spherical mean μ(Z, present), A(M), per-concept resultant (eps-guarded, \|M\|=1 branch) |
| `model/prototypes.py` | Level-2 EMA concept memory {ν_c}: first-epoch-mean init, no-grad EMA (η=0.99) + renorm, DDP all-reduce, staleness reset at warmup end |
| `model/losses_sca.py` | L_sem (row-softmax KL vs S\* + A10 calibration: fixed τ=τ\* or regression, default regression ON), L_align, L_mask (2 toggleable terms), L_concept (eps-floor), L_unif (optional (1−S\*) weighting) |
| `model/lora.py` | LoRA into attention W_q/W_v of the 3 backbones (BERT `query/value`, BEATs `q_proj/v_proj`, EVA-CLIP `q_proj/v_proj` or fused `qkv` slices); per-modality r ∈ {4,8,16}; merge/unmerge |
| `model/pmrl_loss.py` | PMRL baseline head: λ₁ softmax + eigenvalue-concentration (ortho) term; masked raw and /\|M\| variants |
| `data/semantic_targets.py` | S\* cache builder (frozen sentence embeddings over annotation captions; τ\* sharpening; top-k sparsification; disk cache) + batch gather. **The only new data artifact.** |
| `data/mask_sampler.py` | p_full schedule, m† draw (uniform / freq-weighted), zero-fill helper, virtual-mask bookkeeping |
| `evaluation/eval_missing.py` | E4/E5: {0,25,50,75}% × which-modality grid, per-cardinality score stats, rank-displacement bias, per-cardinality affine calibration |
| `evaluation/eval_calibration.py` | E6: S vs S\* regression (R², per cardinality), graded nDCG |
| `utils/volume.py` | + `volume_computation_mean_imputed` — GRAM-masked baseline variant (ii); variant (i) (`volume_computation_masked`) was already present |
| `model/hypergraph.py` | + `concept_incidence(labels)` (Level-2 grouping, existing incidence conventions) |
| `config/sca/` | pretrain + finetune configs; data_cfg blocks inherited **verbatim** from `config/gram` with `${DATA_ROOT}`/`${WORK_ROOT}` parameterization (expanded by `utils/args.py`) |
| `tests/` | §1.4 guards + GRAM regression suite (CPU-only): `python -m pytest tests/ -q` |

Note: the `data/` dataset/loader package (`data/loader.py`, `data_registry`, …) comes from the
upstream GRAM/VAST codebase on the cluster (as in HyperAlign, it is not committed here);
`data/semantic_targets.py` and `data/mask_sampler.py` slot in beside it.

## Running (P2: k=4 pretrain + LoRA)

Compute nodes are OFFLINE — prefetch all network-fetched models once on a LOGIN node into a
directory **outside $HOME** (quota), then every launcher sources the generated `env.sh`
(sets `HF_HOME` + `*_OFFLINE=1`):

```bash
export DATA_ROOT=/path/to/Multimodal_HyperGraph_Dataset   # vast27m_150k, MSRVTT_full, ...
export WORK_ROOT=/path/to/work                            # holds the VAST ckpt

# 0. LOGIN NODE, one-time: prefetch sentence encoders + bert + EVA-giant weight into
#    $WORK_ROOT/sca_models (verifies BEATs + VAST ckpt presence; --with-smoke-data adds
#    CLIP-B/32 + Flickr8k so smoke stage 2 also runs offline)
python3 scripts/prefetch_models.py --models_dir $WORK_ROOT/sca_models --with-smoke-data
export MODELS_DIR=$WORK_ROOT/sca_models

# 1. one-time: build the S* cache (per dataset; ~minutes on one GPU)
python3 data/semantic_targets.py \
  --annotation_json $DATA_ROOT/vast27m_150k/annotations150k.json \
  --out_path $DATA_ROOT/vast27m_150k/s_star_150k.pt

# 2. REQUIRED pre-flight gate (stages 1-2 anywhere; stage 3 = real 24-step k=4 smoke
#    on vast27m_150k with LoRA where $DATA_ROOT is visible)
bash scripts/smoke_test.sh

# 3. pretrain (4xA100). Stage-0 first (heads-only, vision/audio frozen, cheap sanity),
#    then Stage-1 (LoRA r=8 in all three backbones) -- the P2 run:
#    config/sca/pretrain_cfg/sca_pretrain_stage0.json   (Stage-0)
#    config/sca/pretrain_cfg/sca_pretrain.json          (Stage-1, default of the launcher)
sbatch slurm_scripts/run_pretrain_sca.sh

# finetune / eval: same flow as GRAM with the sca configs
python3 run.py --config ./config/sca/finetune_cfg/retrieval-msrvtt.json --output_dir ./workdir/sca_msrvtt
```

k=4 note: the pretrain volume/centroid runs over T+{V,A,S} — subtitles flow from
annotations150k.json (`raw_subtitles`), so no extra config is needed; the val task is
`ret%tvas`. The trunk's `bert.py` targets the cluster env's transformers 4.x pin; the
A10/smoke scripts also run under 5.x (they touch only CLIP + sentence encoders).

`model_cfg.score_mode` selects the eval geometry in `evaluation_mm`:
`volume` (GRAM, default), `centroid` (SCA), `volume_mean_imputed` (masked baseline ii).

## Baselines (P3)

All baselines share the trunk, so comparisons isolate the geometry:

| arm | how |
|---|---|
| GRAM | `model_type: gram` (byte-for-byte intact) |
| GRAM masked-(i) | default eval scoring (`volume_computation_masked`, already per-clip) |
| GRAM masked-(ii) | `score_mode: volume_mean_imputed` (`benchmark_eval/configs_baselines/zs_msrvtt_tvas_maskedii.json`, ckpt via `$EVAL_CKPT`) |
| GRAM + LoRA parity | `model_type: gram_lora` (`config/baselines/pretrain_cfg/gram_lora_pretrain.json`) |
| PMRL head (raw, /\|M\|) | `model_type: pmrl`, `pmrl_variant: raw\|norm`, eval `score_mode: pmrl_raw\|pmrl_norm`; full-FT and LoRA-parity configs + 24-step smoke in `config/baselines/pretrain_cfg/` |
| published rows | `benchmark_eval/published_rows.json` — GRAM paper rows auto-extracted from `make_results_xlsx.py` (`import_published_rows.py --regen/--check`); VAST/ImageBind/LanguageBind/UMT-L/InternVideo2/mPLUG-2/VideoPrism ship as explicit null slots (rendered as dashes, never 0) to be filled from the papers |

## Eval grids (P4): the E4/E5/E6 2×2 design

**Train axis** (which checkpoint):

| method | train-full | train-masked |
|---|---|---|
| SCA | `sca_pretrain_nomask.json` (p_full ≡ 1) | `sca_pretrain.json` (default schedule) |
| GRAM | `config/gram` pretrain (default) | `gram_masked_pretrain.json` (`train_mask: true`) |
| PMRL | `pmrl_pretrain.json` | `pmrl_masked_pretrain.json` |

`train_mask` is a default-off hook in `gram.py` (shared by gram/gram_lora/pmrl): an m†
draw zero-fills gallery features before the loss, so `present_from_feats` trains the
volume at reduced arity — honest masked-(i) training. SCA configs must keep it off (its
own μ_M/μ_K virtual masking; combining would double-mask — guarded at construction).

**Test axis**: `evaluation/run_eval_grids.py` — one encoder pass per checkpoint dumps
features (`--config … --dump_features`), then every scorer × {0,25,50,75}% ×
which-modality grid runs from the same tensors (`--features … --out …`, optional
`--s_star` for the E6 block). Orchestrated over all cells by `scripts/run_e4_grid.sh`
(`CKPTS="name=path …" EVAL_CFG=… bash scripts/run_e4_grid.sh` → `results/e4/*.json`).
Per-cardinality score stats, rank-displacement bias, and the E5 affine calibration
fit/apply are inside each E4 entry.

**Finetune grid**: `slurm_scripts/ft_{msrvtt,didemo,activitynet,vatex,audiocaps}_sca.sh`
+ `ft_msrvtt_depth_sca.sh` (E10: 5-modal `ret%tvasd`, stacks on the finetuned msrvtt
ckpt; the k=5 centroid needs zero code change).

## Ablations + assets (P5)

- **A1–A9 grid**: `config/sca/ablations/` — 26 arms, one config per table row, each a
  delta on the A10-decided base; regenerate with `scripts/gen_ablation_configs.py`
  (`MANIFEST.md` maps arm → config → change → consuming experiment, incl. the A9 S\*
  cache build commands). New knobs: `sca_centroid_gates` (A7 learned-gate centroid,
  zero-init ≡ uniform), `sca_proto_mode: ema|batch` (A4 batch-only ν_c).
- **A2 analysis**: `evaluation/measure_comparison.py` — A(M) vs (1/|M|)·log det G_M vs
  λ₁/|M| per clip (+ pairwise correlations) and the rankings the three scorers induce,
  from any feature dump.
- **E8 diagnostics**: `evaluation/diagnostics.py` — modality gap, RankMe/effective rank,
  Wang-Isola alignment/uniformity, PCA/t-SNE projections (t-SNE requires scikit-learn;
  never silently substituted).
- **LaTeX tables**: `benchmark_eval/make_latex_tables.py` — published rows
  (`published_rows.json`) merged with measured-row JSONs
  (`{"method","section","rows":{"MSR-VTT|T-V":[R1,R10]}}`) into booktabs tables; nulls
  render as dashes, best value per column bolded.

## Guards (encoded as asserts/tests)

- |M|=1 degeneracy: μ = the surviving embedding, A(M)≡1 with the gradient cut.
- Centroid-norm blowup at init: eps guard + L_align-only warmup before L_sem.
- Calibration vs softmax shift-invariance: config **forbids** a learnable τ when a
  calibration mechanism is on (`check_calibration_config`, raises at construction).
- EMA staleness: L_concept delayed to warmup end + prototype reset from the running means.
- GRAM regression: `volume_computation_masked(present=None)` == plain GRAM byte-for-byte,
  all-ones == None, reduced-arity == direct lower-arity volume (`tests/test_sca_units.py`).
