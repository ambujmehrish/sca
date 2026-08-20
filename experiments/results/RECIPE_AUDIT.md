# Recipe audit — is every reported comparison matched?

Triggered by an inconsistency in the tables: Table 1's best SCA row was the lr-1e-4 LoRA
arm while Table 2's best was the lr-2e-5 LoRA arm. Same regime, two learning rates, one
picked per table. This audit traces every reported row back to the config that produced it
and to the recipe published by the method's own authors.

## 1. What the baselines' own papers specify

| method | source | lr | batch | epochs | data |
|---|---|---|---|---|---|
| GRAM | arXiv 2412.11959v2, implementation details | **1e-4** | 256 | 1 | 150k random VAST-27M |
| PMRL | arXiv 2507.17343v1, appendix | **2e-5** | **64** | 1 | VAST-150K |
| HyperGRAM | Na et al., CVPR 2026 | not stated in any source we can reach | -- | -- | 150k |

GRAM's repo README shows `--learning_rate 2e-5` in a *downstream finetuning* example; the
1e-4 above is the *pretraining* (reshaping) stage, which is the stage we reproduce. The two
are different stages and both numbers are correct in their own context.

## 2. What we actually ran

Verified by resolving each config's `default` chain (`scripts/`-side audit, this file's
companion commands):

| arm | lr | batch | epochs | task | matches its paper? |
|---|---|---|---|---|---|
| GRAM (reproduced) | 2e-5 | 256 | 5 | ret%tv%ta | **no — lr should be 1e-4** |
| GRAM + masked / + LoRA | 2e-5 | 256 | 5 | ret%tv%ta | n/a (our variants) |
| PMRL (reproduced) | 2e-5 | 256 | 5 | ret%tv%ta | lr yes; **batch 256 vs their 64** |
| HyperGRAM (reproduced) | 2e-5 | 256 | 5 | ret%tv%ta | unknown |
| SCA (2e-5) | 2e-5 | 256 | 5 | ret%tv%ta | -- |
| SCA (1e-4) | 1e-4 | 256 | 5 | ret%tv%ta | matches GRAM's recipe |

Batch size, epoch setting, training task and initialisation checkpoint are **identical
across every arm** — the learning rate is the only knob that differs, plus `use_lora`,
which is the intended variable.

## 3. Findings

**F1 — our GRAM reproduction is off-recipe.** GRAM pretrains at 1e-4; we retrained it at
2e-5. Severity is limited by an accident of the design: we also evaluate GRAM's *officially
released checkpoint*, which the authors trained at 1e-4, and in our environment it scores
**52.5** against our 2e-5 retrain's **52.4**. The two agree to 0.1, which is evidence that
the learning rate does not move GRAM much in this environment — but it is evidence, not
proof, and the matched-recipe run has not been done. Configs and launcher exist:
`scripts/submit_baselines_lr1e4.sh`.

**F2 — the headline comparison is matched, via the released checkpoint.** SCA at 1e-4
(54.9) against GRAM's released checkpoint at 1e-4 (52.5) is +2.4 at a common learning rate.
The comparison against our own retrain (52.4, +2.5) is the one that is off-recipe.

**F3 — PMRL's batch size is off-recipe** (256 vs their 64). Not fixed; noted as a limitation.

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

1. `bash scripts/submit_baselines_lr1e4.sh` — GRAM, PMRL, GRAM-LoRA at 1e-4. Decides
   whether the headline margin survives a fully matched retrain (F1, F2).
2. Rerun the ablation grid from `config/sca/ablations_lr1e4/` (F4).
3. `sbatch slurm_scripts/depth_control.sh` — cheap, eval-only (F6).
4. State PMRL's batch-size mismatch as a limitation (F3).
