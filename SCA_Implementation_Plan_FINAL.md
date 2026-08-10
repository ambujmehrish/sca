# SCA Implementation Plan — FINAL
## Verified against github.com/ambujmehrish/HyperAlign and Leonardo data paths

> Handoff document for Claude Code. Work directly in the HyperAlign repo (fork of the
> official GRAM/VAST codebase + GatedHGNN additions). Add SCA alongside GRAM; the GRAM
> path must remain byte-for-byte intact (regression-tested with present=None).
> Companion theory/experiment reference: SCA_Final_Research_Plan.pdf (§4-7).

---

## 0. Verified starting position

**Code (from repo inspection):**
- `model/gram.py` (1015 ln): GRAM class on VAST trunk — encoder calls, feature gathering
  (all_gather_with_grad), contra projections, forward_ret with volume softmax + DAM +
  caption losses, `_hg_refine` hook.
- `model/hypergraph.py`: GatedHGNN (<=2 layers, zero-init gates), doc_incidence with
  `present` masking, mutual_knn_adj (train-time semantic edges), semantic_incidence.
- `utils/volume.py`: volume_computation{2,3,4,5} + **volume_computation_masked** +
  **present_from_feats** — the honest reduced-arity GRAM-masked baseline (variant i)
  is ALREADY IMPLEMENTED (phantom-axis identity trick, differentiable, present=None
  == plain GRAM byte-for-byte).
- Full infra: build_dataloader/model/optimizer, distributed, pipeline, sched, save;
  evaluation_mm + classification; benchmark_eval config grid + make_results_xlsx.py;
  slurm_scripts per dataset incl. smoke variants.
- **No LoRA anywhere** (full finetune) -> LoRA is genuinely new.
- Leak-free principle: text never a graph vertex -> carries over: text = query side,
  gallery centroids over {V,A,S,(D)}.

**Data (from config inspection) — IDENTICAL to the SCA experimental plan, zero prep:**
- Pretrain: `$DATA_ROOT/vast27m_150k/` (annotations150k.json, clips/, audios_wav/,
  annotations_val300 + audios_val300). VAST ckpt: model_step_204994.pt (staged).
- Downstream: MSRVTT_full (videos+audios+**depth/all**), DiDeMo, ActivityNet, VATEX,
  AudioCaps, vggsound_5k — all under the same root.
- `benchmark_eval/configs/`: zs_{dataset}_{tv,tva,tvas}.json + vggsound_tav +
  COMBINED — the GRAM table structure as configs. Smoke annos exist.
- Task strings already support **ret%tvasd** (5-modality depth) via
  retrieval-msrvtt_depth.json + ft_msrvtt_depth.sh.
- Current configs hard-code `/leonardo_scratch/large/userexternal/anag0000/...` and
  `/leonardo_work/IscrC_GMEG/anag0000/...` -> **parameterize as $DATA_ROOT / $WORK_ROOT
  env vars in all new config/sca/ files** (portability across allocations).
- **Only new data artifact: the S\* cache** — built from captions in
  annotations150k.json by data/semantic_targets.py -> s_star_150k.pt (one preprocessing
  job; per-downstream-dataset caches likewise from their annotation jsons).

---

## 1. File-level work plan

### 1.1 Reuse unchanged
Backbones (`model/{vision,audio,text}_encoders/`), dataloaders + offline processing,
distributed/init/logger/sched/save/pipeline, evaluation harnesses, benchmark_eval +
xlsx pipeline, slurm launchers, `utils/volume.py` in full (GRAM losses + masked-(i)
baseline + A2 ablation measure).

### 1.2 Modify
| Path | Change |
|---|---|
| `model/gram.py` -> new sibling `model/sca.py` | reuse encoder calls, gathering, projections, config plumbing; swap forward_ret loss block for SCA losses; `model=gram|sca` config switch so all baselines share the trunk |
| `model/hypergraph.py` | untouched (HGNN ablation arm); add `concept_incidence(labels)` helper for Level-2 grouping using existing incidence conventions |
| `config/` | new `config/sca/{pretrain_cfg,finetune_cfg}/` inheriting data_cfg blocks VERBATIM from config/gram (same paths, splits, vision_sample_num, tasks) with $DATA_ROOT substitution; model_cfg adds loss weights (alpha,beta,delta,lambda), p_full schedule, LoRA ranks, eta, tau/tau*, calibration-term toggle |
| `utils/build_model.py`, `args.py` | register sca + LoRA flags |
| `utils/build_optimizer.py` | param groups: LoRA (lr x0.1) vs projections; freeze backbones |

### 1.3 New (all small, closed-form)
1. `model/lora.py` — LoRA into attention W_q,W_v of the 3 backbones; per-modality
   r in {4,8,16}; merge/unmerge.
2. `model/centroid.py` — masked spherical mean mu(Z, present) (eps-guarded), A(M);
   tests: |M|=1 branch (return z, skip A grad), near-antipodal counter,
   present=all-ones == plain mean.
3. `model/prototypes.py` — EMA memory {nu_c}: init from first-epoch class means;
   no-grad EMA (eta=0.99) + renorm; DDP all-reduce of batch means; staleness reset
   at warmup end.
4. `model/losses_sca.py` — L_sem (row-softmax KL vs S*; calibration mechanism per
   A10: fixed tau=tau* OR + cal_w*||S-(2S*-1)||^2, default regression on), L_align,
   L_mask (2 toggleable terms), L_concept (eps-floor on 1-A(c)), L_unif
   (optional (1-S*) weighting).
5. `data/semantic_targets.py` — S* from frozen sentence-embedding affinities over
   annotation captions; tau* sharpening; sparsification; disk cache; batch gather.
6. `data/mask_sampler.py` — p_full schedule (1.0 -> 0.5), m-dagger draw (uniform /
   freq-weighted), zero-fill upstream so present_from_feats sees it; virtual-mask
   bookkeeping (both mu_M and mu_K from one forward pass).
7. `evaluation/eval_missing.py` — grid {0,25,50,75}% x which-modality; per-cardinality
   score stats; rank-displacement bias; per-cardinality affine calibration fit/apply.
8. `evaluation/eval_calibration.py` — S vs S* regression per cardinality; graded nDCG.
9. `utils/volume.py` (+~20 ln) — GRAM-masked variant (ii): mean-imputed missing vector.
10. `model/pmrl_loss.py` — lambda_1 softmax + eigvec orthogonality; masked raw and
    /|M| variants (wrap released PMRL code if importable, else reimplement).

### 1.4 Guards encoded as asserts/tests (from k=2 analysis)
|M|=1 degeneracy; centroid-norm blowup at init (eps + L_align-only warmup before
L_sem); calibration vs softmax shift-invariance (config forbids learnable tau when
calibration claims on); EMA staleness (delay L_concept to warmup end + reset).

---

## 2. Phases

**P1 (wk 1) — Core on k=2.** Items 1.3.2-1.3.6 + `model/sca.py`; image-text smoke
(current experiments = Stage-0 validation); **run A10 first here** (cheap, fixes the
E6 headline config before any 4xA100 run). Wandb: A(M) hist, min||sum z||, S-vs-S*
Pearson, HGNN gates when arm on.
**P2 (wk 2) — k=4 + LoRA.** VAST ckpt via existing path; LoRA; 150k pretrain
(batch 256, 4xA100, clone run_pretrain.sh); Stage-0 then Stage-1.
**P3 (wk 2-3) — Baselines.** GRAM path (config switch) + masked-(i) [exists] +
masked-(ii); PMRL head; LoRA-parity runs of both; import published rows into the
xlsx pipeline.
**P4 (wk 3-4) — Grids.** eval_missing + eval_calibration; 2x2 train/test masking for
all methods; finetune grid via ft_*.sh clones (incl. depth config for E10).
**P5 (wk 4-5) — Ablations + assets.** Grid below; t-SNE/modality-gap/RankMe; xlsx ->
LaTeX tables.

## 3. Experiments
- **E1/E2** Retrieval ZS+FT: MSR-VTT, DiDeMo, ActivityNet, VATEX; T-V/T-VA/T-VAS
  via existing zs_/ft configs; R@1/@10; GRAM rows imported.
- **E3** AudioCaps T2A + VGGSound-5K ZS classification; + PMRL grid on same ckpts.
- **E4 (headline)** Missingness sweep {0,25,50,75}% x which-modality x 2x2
  train/test design, all methods, honest masked baselines.
- **E5** Cross-cardinality calibration: |M|=3 vs 4 score-shift, rank-displacement
  bias, L_mask term-2 effect, affine calibration on/off.
- **E6 (headline)** Semantic calibration: S vs S* R^2 per cardinality; graded nDCG.
- **E7** L_concept payoff vs missing rate; prototype vs text-branch classification.
- **E8** Diagnostics: t-SNE by concept, modality gap, effective rank, align/unif.
- **E9** Efficiency: params/wall-clock/memory LoRA vs GRAM full-FT; rank-vs-quality.
- **E10 (new, free)** 5-modality depth (ret%tvasd, existing msrvtt_depth config):
  centroid needs zero code change for k=5 vs GRAM's volume_computation5 —
  direct arity-invariance demonstration mirroring GRAM Table 4.

## 4. Baselines
GRAM (same trunk; masked-(i) existing; masked-(ii) new; LoRA-parity) | PMRL (head;
masked raw & /|M|; parity) | VAST (starting ckpt ZS rows) | ImageBind / LanguageBind /
UMT-L / InternVideo / mPLUG-2 / VideoPrism (imported rows) | MAP-style prompts
(E4 representative) | SCA-full-FT (parity other direction).

## 5. Ablations
A1 L_mask term-2 off (miscalibration appears in E5). A2 A(M) vs (1/|M|)logdet G_M vs
lambda_1/|M| (reuse volume.py + pmrl head). A3 S*=I vs graded. A4 Level-2: off /
batch-only nu_c / EMA; delta, eta, eps-floor. A5 mask distribution + schedule +
2-drop. A6 Stage-0 vs LoRA r in {4,8,16} vs asymmetric ranks vs full-FT.
A7 uniform centroid vs learned gates vs **GatedHGNN refinement (existing _hg_refine;
semantic edges on/off)** — the Level-3 comparison, free; paper narrative: persistent
EMA prototypes replace train-time-only transductive kNN hyperedges.
A8 lambda sweep incl. 0; S*-weighted repulsion. A9 S* source/tau*/sparsification.
A10 calibration mechanism (KL-only vs KL+regression vs fixed-tau) — FIRST, on k=2.

## 6. Compute (Leonardo A100, Class-C)
P1: single A100, hours. Pretrain ~12 configs x (1 epoch, 150k, 4xA100) ~= 1.5-2k
A100-h. FT + eval grids ~= 1-1.5k. Total ~= 3-4k A100-h. LoRA runs cut optimizer
memory vs full-FT — record for E9.

## 7. Definition of done
1. E1-E3 xlsx with SCA rows next to imported GRAM rows (identical splits guaranteed
   by verbatim data_cfg inheritance).
2. E4 curves: SCA degrades more gracefully than masked-GRAM(i/ii) and masked-PMRL;
   E5 near-zero cardinality bias with term-2 on.
3. E6 calibration R^2 under the A10-decided config; baselines shown uncalibrated by
   construction.
4. A2 answers first-order sufficiency either way (both outcomes shape a publishable
   framing).
5. E10 runs with zero centroid-code change at k=5.
6. Headline numbers x3 seeds; one YAML per table row; GRAM path regression test
   (present=None byte-for-byte) in CI; all new configs use $DATA_ROOT/$WORK_ROOT.
