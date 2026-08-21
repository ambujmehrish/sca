# Recipe audit — is every reported comparison matched?

Triggered by an inconsistency in the tables: Table 1's best SCA row was the lr-1e-4 LoRA
arm while Table 2's best was the lr-2e-5 LoRA arm. Same regime, two learning rates, one
picked per table. This audit traces every reported row back to the config that produced it
and to the recipe published by the method's own authors.

## 1. What the baselines' own papers specify

| method | source | lr | batch | epoch field | data |
|---|---|---|---|---|---|
| GRAM | **their released config** (`config/gram/default_run_cfg.json` + `finetune_cfg/pretrain-gram.json`) | **1e-4** | **128** | 5 | 150k VAST-27M |
| GRAM | their paper (arXiv 2412.11959v2) | 1e-4 | 256 | "single epoch" | 150k random |
| PMRL | their paper (arXiv 2507.17343v1, appendix) | **2e-5** | **64** | one epoch | VAST-150K |
| HyperGRAM | Na et al., CVPR 2026 | not stated in any source we can reach | -- | -- | 150k |

Three things worth recording:

- GRAM's repo README shows `--learning_rate 2e-5`, but that is the *downstream finetuning*
  example. The pretraining (reshaping) stage we reproduce is 1e-4. Both are correct in
  their own context, and this is the likeliest origin of our 2e-5.
- GRAM's paper and GRAM's config disagree with each other on batch size (256 vs 128) and on
  epochs ("single epoch" vs `epoch: 5`). We reproduce **the config**, since that is what
  they actually ran.
- Our `config/sca/default_run_cfg.json` is byte-identical to GRAM's released
  `default_run_cfg.json` apart from one added `save_steps` key -- same `clip_lr` 5e-7, same
  betas, weight decay, grad-norm clip, warmup ratio, fp16, seed 50. **1e-4 was therefore
  already the inherited default, and the 2e-5 in our per-arm configs is an explicit
  override we added.** The deviation was one line, not a different pipeline.

## 2. What we actually ran

Verified by resolving each config's `default` chain (`scripts/`-side audit, this file's
companion commands):

| arm | lr | batch | epochs | task | matches its paper? |
|---|---|---|---|---|---|
| GRAM (reproduced) | 2e-5 | 256 | 5 | ret%tv%ta | **no — should be lr 1e-4, batch 128** |
| GRAM + masked / + LoRA | 2e-5 | 256 | 5 | ret%tv%ta | n/a (our variants) |
| PMRL (reproduced) | 2e-5 | 256 | 5 | ret%tv%ta | lr yes; **batch 256 vs their 64** |
| GRAM `_paper` (wave 9) | 1e-4 | 128 | 5 | ret%tv%ta | yes |
| PMRL `_paper` (wave 9) | 2e-5 | 64 | 5 | ret%tv%ta | yes |
| HyperGRAM `_paper` (wave 9) | 1e-4 | 128 | 5 | ret%tv%ta | assumed = GRAM's |
| HyperGRAM (reproduced) | 2e-5 | 256 | 5 | ret%tv%ta | unknown |
| SCA (2e-5) | 2e-5 | 256 | 5 | ret%tv%ta | -- |
| SCA (1e-4, "T1") | 1e-4 | 256 | 5 | ret%tv%ta | lr yes; **batch 256 vs GRAM's 128** |
| SCA `_paper` (wave 9) | 1e-4 | 128 | 5 | ret%tv%ta | yes |

Within the first-pass generation, batch size, epoch setting, training task and
initialisation checkpoint are identical across every arm, so those comparisons differ only
in learning rate and `use_lora`. Against the *published* recipes, however, both the batch
size and the learning rate were off — see F1/F3.

## 3. Findings

**F1 — our GRAM reproduction is off-recipe.** GRAM pretrains at 1e-4; we retrained it at
2e-5. Severity is limited by an accident of the design: we also evaluate GRAM's *officially
released checkpoint*, which the authors trained at 1e-4, and in our environment it scores
**52.5** against our 2e-5 retrain's **52.4**. The two agree to 0.1, which is evidence that
the learning rate does not move GRAM much in this environment — but it is evidence, not
proof, and the matched-recipe run has not been done. Configs and launcher exist:
`scripts/submit_baselines_lr1e4.sh`.

**F2 — the headline comparison is matched on lr only, not on batch size.** SCA at 1e-4
(54.9) against GRAM's released checkpoint (52.5) shares a learning rate, but SCA trained at
batch 256 where GRAM's recipe is 128. Batch size sets the number of in-batch contrastive
negatives, which a centroid objective is directly sensitive to, so this is not a free
parameter. `sca_paper` (lr 1e-4, batch 128) is the first genuinely matched SCA row, and
**the margin may not survive it** — GRAM at its own recipe may also score above the 52.4/52.5
we have measured so far. That is the point of running it.

**F3 — PMRL's batch size is off-recipe** (256 vs their 64), and GRAM's is too (256 vs
their 128). Both are fixed by the wave-9 `_paper` configs.

**F4 — every ablation arm is at the wrong learning rate.** All 26 A1–A9 configs are deltas
on the lr-2e-5 config, so they ablate components of a model that is not the one Table 1
reports. Eight of them already have measured results (A5 p03, A6 asym/r16/fullft, A7 gates,
A8 λ=0.05/0.3/weighted). If the paper reports SCA at 1e-4, the whole grid has to be rerun
on that base. `scripts/gen_ablation_configs.py` now takes `SCA_ABLATION_BASE` /
`SCA_ABLATION_OUT`; `config/sca/ablations_lr1e4/` holds the regenerated 1e-4 grid.

**F5 — seed averaging was inconsistent between columns (fixed).** R@1 was a 3-seed mean
while R@10 was the seed-42 value alone. Corrected in Tables 1 and 8:

| row | column | was | now |
|---|---|---|---|
| GRAM$^\dagger$ | V2T R@10 | 80.9 | 79.8 |
| SCA (1e-4) | T2V R@10 | 83.5 | 83.3 |

**F6 — the depth row is not a controlled comparison** (found separately, already recorded
in Table 3's caption): it continues the MSR-VTT finetune for four more epochs at 1e-4, so
its +1.6 confounds arity with the extra pass. Control: `slurm_scripts/depth_control.sh`.

## 4. What is NOT wrong

- No measured value was ever edited. Every number in the final tables resolves to a row
  JSON under `experiments/results/*/rows/`.
- Budget, batch size, epochs, training task and init checkpoint are identical across all
  families, so the comparisons differ only in the knobs named above.
- The metric is consistently the ITM-reranked one from `wave2` onward. `wave1` row files
  hold raw-scorer numbers under the same method names — a naming collision to be careful
  with when scripting over the rows directory, but the tables use the ITM values.

## 5. Order of work

All of it is one parallel submission -- `bash scripts/submit_recipe_runs.sh` (25 jobs):

1. **Baselines at their authors' recipes** (F1, F2, F3) -- `gram_paper`, `pmrl_paper`,
   `gram_lora_paper`, `gram_hyp_paper` -- **and SCA under the same recipe**, `sca_paper` /
   `sca_paper_fullft` (lr 1e-4, batch 128). Together these decide whether the headline
   margin survives a fully matched comparison; no existing SCA arm is matched on batch size.
2. **Ablation grid on the reported SCA configuration** (F4) -- 21 arms from
   `config/sca/ablations_lr1e4/`; the five A9 arms wait on their S* caches.
3. `sbatch slurm_scripts/depth_control.sh` -- cheap, eval-only (F6).

### Preflight

`python3 scripts/preflight_runs.py [--phase headline|baselines|ablations|all]` drives the
submitter's `--dry` output and refuses to pass unless every config parses, every workdir /
job name / log pattern is unique, no workdir pre-exists unstamped, and — the check that
earns its keep — **no two arms resolve to the same config**. It caught `sca_paper_fullft`
and `A6_full_ft` resolving byte-identically: the same full pretrain queued twice under two
names. `A6_full_ft` is now dropped from Phase 3 and the ablation table reads its
full-finetuning row off the Phase-1 result.

### No two runs overlap, and no run rewrites a config

- **Configs are read-only at runtime.** The only config written during a run is
  `<workdir>/log/hps.json` (`utils/args.py`), inside that run's own output directory. No
  code path writes back to `config/`, so 30 parallel jobs reading the same
  `default_run_cfg.json` cannot contaminate each other — verified by grep over every
  `json.dump` / file write in `run.py` and `utils/`.
- **Exclusive lock per workdir.** `run_config.sh` takes an atomic `mkdir` lock on
  `<workdir>/.lock` before touching anything, recording the owning job id.
  `--dependency=singleton` only protects jobs submitted with the same `-J`; the lock also
  covers a hand-submitted `sbatch` or a job-name typo. A lock whose owner is no longer in
  `squeue` is reclaimed with a warning, and the trap releases it on EXIT/TERM/INT so a
  wall-clock timeout does not block the next resubmission. Tested both directions: blocks
  while the owner is alive, reclaims when stale.
- **Post-run verification.** `python3 scripts/verify_runs.py [--phase ...]` reads each
  workdir's `log/hps.json` — the resolved options as the job actually used them — and
  compares lr, batch size, model type, LoRA on/off, rank and alpha against the config that
  workdir was supposed to use, plus the `.provenance` fingerprint. Statuses: OK / MISMATCH /
  NOT STARTED / NO STAMP, exit 1 on any mismatch. Run it when the first checkpoints appear
  and again before extracting into tables. Preflight proves the *plan*; this proves the
  *outcome*.

### Naming and isolation

Each job is submitted with `-J <arm> -o slurm_scripts/logs/<arm>_%j.out`, so logs are
`gram_paper_1234567.out`, not thirty anonymous `run_*.out` files. Checkpoints stay in that
arm's own `workdir_pretrain/<arm>/ckpt`. `--dependency=singleton` means a resubmission for
continuation (the 6h wall clock needs several) queues behind the running job instead of
starting a second process into the same checkpoints, and the rendezvous port is derived from
`$SLURM_JOB_ID` rather than `$RANDOM`, which with 30 jobs in flight was a live collision.

### Keeping the two generations apart

Every wave-9 run writes to a **new** workdir (`<method>_paper`, `abl1e4_<arm>`); no
first-pass directory is reused, so both generations stay independently extractable. On top
of that, `slurm_scripts/run_config.sh` now stamps each workdir with a SHA-256 fingerprint of
its *resolved* config chain plus CLI args, and refuses to resume when the stamp disagrees:

```
FATAL: workdir_pretrain/gram_paper was created by a DIFFERENT config -- refusing to mix runs.
       stamped: config/baselines/pretrain_cfg/gram_pretrain.json (fingerprint 22e25ef7af826ae1)
       asked:   config/baselines/pretrain_cfg/gram_paper.json    (fingerprint cd7467ea12acd3a1)
```

Verified to distinguish lr overrides, `--seed` values, and edits to an *inherited* default
file. Previously the script resumed from whatever checkpoints happened to be in the output
directory, with no check of their origin -- that was the mechanism by which two arms could
silently become one.

When extracting, give wave-9 rows distinct method names (`GRAM (paper recipe)`, not `GRAM
(repro)`): `wave1` and `wave2` already collide on method name with different metrics, and
`scripts/extract_results.py` keys rows by name.

## F6 — the published protocol, read from the paper source (2026-08-20, corrected)

Downloaded arXiv:2412.11959v2 and read Table 5 and the body text directly. An earlier pass
today relayed a summarizer's reading of the same table and got the frame counts wrong; the
verbatim table is reproduced here so it does not happen again.

**Table 5 (Appendix B.1).** Columns are Train / Val / Test counts, then "# Frames", then
"# Epochs". Caption: *"# Frames refers both to training and inference."*

| Benchmark | Train | Val | Test | # Frames | # Epochs |
|---|---|---|---|---|---|
| AudioCaps | -- | -- | 700 | 8 | -- |
| VGGSound | -- | -- | 5000 | 8 | -- |
| DiDeMo | 8394 | 1065 | 1003 | 8 | 40 |
| ActivityNet | 10009 | -- | 4917 | 8 | 20 |
| MSR-VTT | 9000 | -- | 1000 | 8 | 4 |
| VATEX | 14060 | -- | 431 | 8 | 3 |

**Body text, verbatim:** *"We pretrain the GRAM-based model on a subset of the VAST27M
dataset comprising 150k random samples with a learning rate of 1e-4 using the AdamW
optimizer with weight decay and batch size of 256. For finetuning we reduce the batch size
to 64 and change the number of epochs according to the specific dataset, the complete
details are shown in Tab. 5."*

| item | paper | ours | status |
|---|---|---|---|
| pretrain lr / batch | 1e-4 / 256, 1 epoch | T1: 1e-4 / 256 | matches |
| finetune batch | 64 | 64 | matches |
| frames, train and inference | 8, every benchmark | 8 | **correct as it was** |
| VATEX test split | 431 (14060 train + 431 = the 14491 retained) | 431 | matches |
| MSR-VTT / DiDeMo / ActivityNet test | 1000 / 1003 / 4917 | to verify on cluster | open |

**Corrections to earlier entries in this file.**

1. The "40 inference frames on DiDeMo, 20 on ActivityNet" claim was wrong. Those are
   *finetuning epochs*. 22 eval configs were changed on that basis and have been reverted
   to 8 frames. `tests/test_eval_protocol.py` now asserts 8 and documents why.
2. VATEX 431 is GRAM's own test split, not a reduced subset of ours: 14060 train + 431
   test is exactly the 14491 samples the paper says it retained out of 41250. Our column
   is comparable, and the "audio-complete subset, not comparable" caveat carried since
   wave 3 was never correct.
3. The surviving real defect is `max_caption_len`: every DiDeMo/ActivityNet config that
   shipped with this repo sets 70, and the configs generated for this campaign inherited
   the default 40. That is a codebase convention rather than a statement in the paper, but
   it is GRAM's own convention for those two datasets and the truncation is real. Fix
   stands; those two benchmarks need re-evaluation.

## F8 — one configuration, not a per-benchmark best (2026-08-20)

Requirement: SCA's reported numbers must all come from a SINGLE (learning rate, batch
size), not from picking whichever arm wins each benchmark. Batch is already single at 256
for both SCA arms, so the open choice is the learning rate: 2e-5 (better transfer) versus
1e-4 (better MSR-VTT). Neither dominates on today's numbers.

That choice cannot be made yet, because DiDeMo and ActivityNet are currently measured
under the caption truncation above. Sequence: fix (done) -> re-evaluate BOTH SCA arms on
all five benchmarks -> pick the single lr with the best overall standing -> report only
that arm everywhere. The losing arm moves to the ablation table as a learning-rate row.

## F9 — read from source: zero-shot vs finetuned, and who trains what (2026-08-20)

Downloaded both papers and read the tables directly.

**Neither GRAM nor PMRL trains from scratch.** Both start from the released VAST
checkpoint and continue-pretrain on a 150k subset of VAST-27M for one epoch, then report
downstream results in two separate settings.

- GRAM: *"Starting from VAST pretraining models, we further pretrain those on a small
  subset of VAST27M comprising 150k samples using our defined loss functions … We set the
  batch size to 256 and a single epoch pretraining on 4 NVIDIA A100 cards."*
- PMRL: *"PMRL is built upon VAST and employs a continual pre-training strategy … we
  utilize VAST-150K to re-boost its zero-shot capabilities, and split downstream datasets
  for fine-tuning PMRL for specific tasks."*

This is exactly our setup, so the comparison is structurally sound. The only "from
scratch" training in either paper is GRAM's loss ablation (their Tab. 6) and PMRL's
random-init analysis on the ABIDE task — neither is a retrieval headline number.

**Zero-shot and finetuned are separate tables in both papers.** GRAM: Tab. 1 zero-shot,
Tab. 2 finetuning (appendix Tabs. 8/9 and 10/11 add R@10). PMRL: Tab. 1 zero-shot, Tab. 2
finetuning. Our transcriptions come from the ZERO-SHOT tables and are modality-matched:

| our column | source | verbatim |
|---|---|---|
| GRAM MSR-VTT T-VAS | GRAM Tab. 1, T-VAS row | 54.8 / 52.9 |
| GRAM DiDeMo, ActivityNet T-VA | GRAM Tab. 1, T-VA row | 54.2 / 52.2, 59.0 / 50.4 |
| GRAM VATEX T-VAS | GRAM Tab. 1, T-VAS row | 83.5 / 82.7 |
| PMRL all four | PMRL Tab. 1 | 54.5/52.4, 50.6/48.4, 56.0/49.6, 80.5/75.2 |

For contrast, GRAM's FINETUNED numbers (their Tab. 2) are much higher -- MSR-VTT T-VAS
64.0, DiDeMo T-VA 67.3, ActivityNet T-VA 69.9 -- so a zero-shot row must never be set
against those.

**GRAM measured by three independent parties.** PMRL's Tab. 1 includes its own
re-evaluation of GRAM, and it lands far below GRAM's self-report -- much closer to ours:

| GRAM, T2V R@1 | MSR-VTT | DiDeMo | ActivityNet | VATEX |
|---|---|---|---|---|
| GRAM's own paper | 54.8 | 54.2 | 59.0 | 83.5 |
| PMRL's reproduction | 51.5 | 49.8 | 54.5 | 77.5 |
| our reproduction | 52.4 | 49.6 | 52.0 | 88.9 |

On DiDeMo our reproduction and PMRL's agree to 0.2, and on ActivityNet both sit well below
GRAM's 59.0. So the "we are 7 points cold on ActivityNet" worry is substantially a property
of GRAM's published numbers, not of our pipeline: an independent group also fails to
reproduce them. This is worth a sentence in the paper, citing PMRL's table.

VATEX is the exception -- ours is 11 points above PMRL's GRAM row, which suggests PMRL's
VATEX split differs from the 431-clip one GRAM and we use. Their paper does not state a
VATEX test size, so that column should not be cross-compared with PMRL.

## F10 — AudioCaps finetuning was train-on-test (2026-08-20)

`config/{sca,gram}/finetune_cfg/retrieval-audiocaps.json` pointed **both** the training
split and the validation split at `benchmark_eval/audiocaps_tva_annotation.json` — the
704-clip AudioCaps test annotation — with `training: true`. Any number it produced was
trained on the data it was scored on.

The affected result is **SCA ft AudioCaps 51.6 / 50.6**, now moved to
`experiments/results/quarantine/`. It was never in Tables 1–9, only in the auto-generated
wide `finetune_t2v.tex` / `finetune_v2t.tex`, so nothing in the paper's table set depended
on it.

The config came from the imported trunk and GRAM never uses it: their Tab. 5 gives
AudioCaps no finetuning epochs, and their AudioCaps numbers (Tab. 3) are zero-shot. So
there is no baseline to compare a finetuned AudioCaps row against, and no protocol reason
to have one. Both configs are deleted rather than repaired — repair would require a real
AudioCaps train split that neither this repo nor the published protocol uses.
`slurm_scripts/ft_audiocaps_sca.sh` now refuses to run and explains why, and
`tests/test_eval_protocol.py::TestNoTrainTestOverlap` fails if any finetuning config
reintroduces an overlap between its train and val annotations.

### Finetuning data, verified per benchmark

| config | train annotation | eval annotation | overlap |
|---|---|---|---|
| MSR-VTT | `msrvtt/descs_ret_train.json` | `msrvtt/descs_ret_test.json` | none |
| MSR-VTT depth | `msrvtt/descs_ret_train.json` | `msrvtt/descs_ret_test.json` | none |
| DiDeMo | `didemo/descs_ret_train.json` | `didemo/descs_ret_test.json` | none |
| ActivityNet | `activitynet/descs_ret_train.json` | `activitynet/descs_ret_test.json` | none |
| VATEX | `vatex/descs_ret_train_aug.json` | `vatex/descs_ret_test_431.json` | none |
| AudioCaps | — | — | **deleted** |

## F11 — annotation audit on the cluster (2026-08-20)

`scripts/audit_annotations.py` run where the data lives. Nine files checked against GRAM's
Tab. 5 counts:

| split | ours | GRAM Tab. 5 | verdict |
|---|---|---|---|
| DiDeMo train / test | 8394 / 1003 | 8394 / 1003 | exact |
| ActivityNet train / test | 10009 / 4917 | 10009 / 4917 | exact |
| MSR-VTT train / test | 9000 / 1000 | 9000 / 1000 | exact |
| VATEX test | **431** | **431** | exact |
| AudioCaps test | 704 | 700 | +4, within tolerance |
| VATEX train (`descs_ret_train_aug.json`) | **26,681** | **14,060** | **1.9x larger** |

No train/eval clip-id overlap anywhere.

**The VATEX training split is the one defect.** 26,681 unique clips is more than the 14,491
GRAM retained across *all* splits — our copy of VATEX lost far fewer videos to takedowns
than theirs did. The consequence is split cleanly by setting:

- **Zero-shot VATEX is comparable.** The test split is 431, matching exactly, and zero-shot
  never touches the train split. Table 2's VATEX column stands as measured.
- **Finetuned VATEX is not comparable.** Our 94.2 / 91.0 was finetuned on ~1.9x the videos
  GRAM used for their 87.7 / 84.2. That advantage is data, not method, and the row must not
  be reported as a win. Either drop it, or subsample the train split to 14,060 clips and
  re-finetune.

This also retires the earlier hypothesis that our VATEX numbers were inflated by an easier
gallery: the gallery is identical. The inflation, where it exists, is on the training side
and only affects the finetuned row.

## F12 — GRAM's VATEX annotations, fetched from their repo (2026-08-20)

Their repo does ship annotations, at `datasets/annotations/vatex/`. Downloaded and counted:

| file | entries | unique clips |
|---|---|---|
| `descs_ret_train.json` (theirs) | 259,910 | **25,991** — the full VATEX train split |
| `descs_ret_test.json` (theirs) | 1,500 | **1,500** — the standard VATEX test split |
| `descs_ret_train_aug.json` (ours) | — | **26,681** |
| `descs_ret_test_431.json` (ours) | 431 | **431** |

Train/test overlap in their files: zero.

This separates two effects that F11 had conflated.

1. **Our train annotation reaches outside their train split.** 26,681 against their 25,991
   — roughly 690 clips that are not in GRAM's VATEX train file at all, most likely VATEX
   val. This is exactly correctable: intersect with their published roster. The ids are now
   committed as `gram_repo_train_ids.txt` and `gram_repo_test_ids.txt`, and
   `scripts/make_vatex_matched_split.py` intersects before doing anything else.
2. **Download attrition is not correctable.** Their Tab. 5 reports 14,060 train and 431
   test — what they could actually fetch from 25,991 and 1,500. Which clips those were is
   not published, so a size-matched split removes the volume advantage but not the sampling
   difference.

**The test split needs attention.** Their repo's VATEX test is the standard 1,500 clips;
their Tab. 5 says they evaluated 431 of them; our file has exactly 431. The count agreeing
is suggestive but not proof that it is the same 431 — ours came from the HyperAlign trunk,
not from GRAM. Until that is established, the VATEX comparison in Table 2 rests on an
assumption, and the honest options are to say so, or to evaluate on the full 1,500-clip
standard split where no such assumption is needed.

## F13 — which GRAM checkpoint the tables use (2026-08-20)

`GRAM_pretrained_TVAS/ckpt/model_step_459.pt` (wave1/validation_official_gram.md), the
4-modality T-VAS checkpoint — `GRAM_pretrained_4modalities` in their release. Correct for
every T-VAS row in Tables 1 and 2. Their `GRAM_pretrained_5modalities` release would only
be needed for a depth/k=5 baseline, which we do not report.

## F14 — depth (k=5): GRAM published a baseline and released the checkpoint (2026-08-20)

GRAM's Tab. 4, zero-shot MSR-VTT T2V R@1 by arity:

| modalities | R@1 |
|---|---|
| T-V | 52.8 |
| T-V-A | 54.1 |
| T-V-A-S | 54.8 |
| **T-V-A-S-D** | **55.3** |

So **their depth gain is +0.5**, and `GRAM_pretrained_5modalities` in their release is the
checkpoint that produced it. Two consequences.

**The +1.6 depth claim can be replaced with a controlled one.** Tab. 3's depth row came
from a finetuned run that continued the row-1 finetune for four more epochs at a different
learning rate, so it confounded arity with extra training — and it had no GRAM baseline.
`slurm_scripts/depth_zeroshot.sh` instead scores one checkpoint per method at 4 and at 5
modalities, zero-shot, nothing trained, and the claim becomes a comparison of deltas
against their published +0.5. The launcher refuses to run without `GRAM5_CKPT`: silently
falling back to the 4-modality checkpoint would score a model that has never seen depth and
report it as GRAM's depth result.

**Their paper is internally inconsistent by 0.1 on one cell.** Tab. 1 gives GRAM T-VA on
MSR-VTT as 54.2; Tab. 4 gives the same configuration as 54.1. We keep 54.2 (Tab. 1, the
main results table) and note the discrepancy so nobody later "corrects" it.

## F12 — implementation audit of the SCA model (2026-08-20)

Prompted by the zero-shot results being incremental rather than by any single symptom.
Checked, and CORRECT:

- **Scorer selection.** `default_model_cfg.json` sets `score_mode=centroid`; the eval logs
  confirm every SCA cell ran `mode=centroid` and every GRAM cell `mode=volume`.
- **`max_caption_len`.** 70 on DiDeMo/ActivityNet, confirmed from each run's `hps.json`.
  The truncation is fixed and did not move the numbers, so our queries never exceeded 40
  tokens and that thread is closed.
- **Presence masking.** Both `model/sca.py` and `evaluation/evaluation_mm.py` derive
  `present` from the embedding norm, so a zero-filled modality is excluded rather than
  averaged in, in training and in evaluation alike.
- **Gallery composition.** Text is excluded from the centroid in both paths.

One asymmetry found, and it is the strongest remaining lead:

**The task string selects the modality set at EVAL but is ignored at TRAINING.**
`evaluation_mm.py:275` takes `_mods = _task[1:]`, so `ret%tvas` scores a {v,a,s} centroid.
`model/sca.py::_gallery_feats` ignores `task` and returns every modality in the batch.

The consequence is not the modality set -- it is the objective. `model/gram.py:519` shows
GRAM's `forward_ret` looping over sub-tasks, so `ret%tv%ta` trains a text-video volume AND
a text-audio volume as two separate objectives. SCA's override collapses that into one
centroid over {v,a} and one loss, so **SCA never trains a dedicated text-video alignment**:
video is always blended with audio before meeting the text.

That predicts the observed benchmark pattern. SCA's margin over the released GRAM
checkpoint tracks how informative audio is -- AudioCaps (T->A 25.1) +3.0, VATEX (15.1)
+0.3, DiDeMo (4.1) -0.7, ActivityNet (3.0) -3.7. Where audio is near-noise the blended
centroid is poor and there is no separately-trained text-video pathway to fall back on;
GRAM has one.

Not yet established: whether the virtual mask already supplies this signal. With
`mask_n_drop=1` on a two-modality gallery the masked view is {v} or {a} alone, which is
exactly the sub-task objective -- but `p_full` starts at 1.0, so early training sees only
full views, and `L_align` is computed on the masked view only. Measuring how often a
single-modality view is actually drawn, and how late, is the next diagnostic and needs no
GPU.

### F12b — quantified: SCA trains text-video on 20.8% of steps, GRAM on 100%

`scripts/diag_mask_schedule.py` replays the sampler over the real run (2925 steps of batch
256 = 150k clips x 5 epochs, 2-modality gallery):

| modality set the centroid sees | clip-steps | share |
|---|---|---|
| full {v,a} | 437,844 | 58.5% |
| **video only** | 155,421 | **20.8%** |
| audio only | 155,535 | 20.8% |

And it is back-loaded — the video-only share is 1.8% through the first 5% of the run, 9.1%
by 25%, reaching 20.8% only at the end. GRAM's `ret%tv%ta` trains its text-video volume on
every step from step 0.

What the existing schedule arms would give: `A5_pfull_end_0.3` 29.0%, `A5_pfull_const_0.5`
25.0%, `A5_mask_2drop` 20.8% (unchanged, since with a 2-modality gallery dropping 2 leaves
nothing and the sampler clamps). A flat `p_full=0.5` from step 0 gives 25.0% and starts at
step 0 instead of step 1.

So no schedule reachable from the current knobs gets past ~29%. Closing the gap to GRAM's
100% means adding the sub-task objectives explicitly -- computing `L_align` against the
{v}-only and {a}-only centroids alongside the full one, which is what `ret%tv%ta` already
asks for and what `_gallery_feats` currently discards. That is a training-loop change using
machinery already present (the same masked centroid, different subsets), not new model
capacity.

Worth measuring before committing to it: this predicts the benchmark ordering but does not
prove it. The cheap test is an arm at `A5_pfull_end_0.3` (29.0%, one pretrain) -- if
ActivityNet moves in proportion, the mechanism is confirmed and the explicit sub-task
version is worth the larger change; if it does not move, the hypothesis is wrong and the
deficit is elsewhere.

## F13 — the 150k subset is really 136,674 clips (2026-08-20)

Measured on the cluster against `$DATA_ROOT/vast27m_150k`:

| | count |
|---|---|
| clips in `annotations150k.json` | 150,154 |
| with video on disk | 136,694 |
| with audio on disk | 136,674 |
| **with both (what trains)** | **136,674** |

13,480 clips (9.0%) are no longer downloadable, so every arm here continue-pretrains on
91% of the nominal budget. That is also why the training-completion audit misfired twice:
the trainer schedules against `len(dataset)`, which is built from what exists, so 2649
steps is five full epochs of 136.7k clips and not a truncated run of 150k.

**Belongs in the setup section as a measured number.** But three qualifications:

1. *Probably symmetric.* GRAM documents the same attrition for VATEX -- "a large part of
   the dataset is now unavailable online due to removed or private videos" -- so they were
   subject to it on VAST-27M too; they simply never report the pretraining subset's
   effective size. Our clips went offline later, so we likely hold fewer, by an unknowable
   margin.
2. *Too small to explain the deficit.* 9% fewer unique clips over five epochs is worth a
   few tenths of an R@1, not the 5-6 points SCA gives back at the ITM stage. It would also
   depress the raw scorer, where SCA wins.
3. *Internally fair.* Our GRAM and PMRL reproductions trained on the identical reduced set,
   so only comparisons against the released checkpoint and the published numbers are
   touched.

### Repo hygiene, found while checking this

`.gitignore:21` is `data/*.py`, so the dataset package -- `data/loader.py`,
`data/__init__.py`, the `annoindexed` class -- is untracked; only `mask_sampler.py` and
`semantic_targets.py` were force-added. A fresh clone cannot train, the data pipeline
cannot be reviewed, and questions like "where are unavailable clips dropped" cannot be
answered from the repository. Worth fixing before release.
