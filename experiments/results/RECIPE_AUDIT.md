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

## F6 — the published protocol, read from the paper (2026-08-20)

Fetched GRAM (arXiv:2412.11959v2) directly rather than inferring from their released
config. Three corrections, two of them to our evaluation:

| item | paper says | we had | effect |
|---|---|---|---|
| pretrain batch | **256**, lr 1e-4, 1 epoch, 4×A100 | T1: 256 @ 1e-4 | **already matched** |
| DiDeMo inference frames | **40** (8 for training) | 8 | under-sampled 5× |
| ActivityNet inference frames | **20** (8 for training) | 8 | under-sampled 2.5× |
| VATEX evaluation set | **14,491 samples** | 431 | not the same task |

1. **Batch 256 is the published recipe.** Our earlier reading of "batch 128" came from a
   released config file, not the paper. SCA T1 (lr 1e-4, batch 256) therefore *already*
   matches GRAM's published pretraining recipe exactly. `sca_paper` at batch 128 is an
   exploratory arm, not a correctness fix.
2. **Frames.** Table 5 (Appendix B.1) gives per-dataset inference frame counts. We
   evaluated DiDeMo and ActivityNet at 8 frames instead of 40 and 20. Combined with the
   `max_caption_len` 40-vs-70 truncation found the same day, our two weakest benchmarks
   were being scored under two independent protocol deviations, both of which hurt every
   method we measure. 22 eval configs corrected; `tests/test_eval_protocol.py` guards both.
3. **VATEX.** They evaluate on 14,491 samples; our annotation has 431. Retrieval over a
   431-item gallery is a much easier problem, so our 90.3 is not comparable to their 83.5
   in either direction. The column is now excluded from bolding and drawn no conclusion
   from. Fixing it properly means obtaining a VATEX test set of comparable size.
