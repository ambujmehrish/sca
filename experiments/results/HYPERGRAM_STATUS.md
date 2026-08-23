# HyperGRAM reproduction — current status

**Read this before quoting any HyperGRAM reproduction number.** This question has been
re-litigated repeatedly from stale records; this file is the single place the answer lives.

## The three arms, and which recipe each used

| arm | hyperbolic branch consumes | lr | batch | MSR-VTT ITM best / final |
|---|---|---|---|---|
| `gram_hyp` (v1) | pre-normalisation projections | 2e-5 | 256 | 35.2 / 35.2 |
| `gram_hyp2` (v2) | the L2-normalised features GRAM uses | **2e-5** | 256 | 51.0 / **37.4** |
| `gram_hyp_paper` (v3) | same as v2 | **1e-4** | 128 | **NEVER TRAINED** |

## Why the 37.4 must not be quoted

`gram_hyp2` was trained at **lr 2e-5**. That rate was inherited from the HyperAlign trunk, and
`wave4/ANALYSIS.md` §1 identified it as the recipe defect for *our own* method: "The recipe gap
was the LEARNING RATE: our 2e-5 was inherited from HyperAlign, but GRAM's published 54.8 was
trained at 1e-4." Moving SCA to 1e-4 took it from 53.5 to 54.9.

The same correction was never applied to the HyperGRAM arm before its number was recorded. So
`gram_hyp2`'s 51.0 → 37.4 collapse is a run at a learning rate the method's own paper does not
use, against a baseline family we had already shown to be learning-rate sensitive. It is not
evidence about HyperGRAM and must not be cited as such — not as a table row, not as an appendix
number, and not as grounds for saying the reproduction "does not work".

`config/baselines/pretrain_cfg/gram_hyp_paper.json` is v2 at lr 1e-4, batch 128 — the published
recipe. It differs from `gram_hyp2_pretrain.json` in exactly those two keys and nothing else.

## What is true right now

- We have **no valid HyperGRAM reproduction**. Not a failed one — an unrun one.
- The published numbers (MSR-VTT 56.6, DiDeMo 51.3, ActivityNet 58.2, VATEX 79.9) are what the
  main table cites, and that stays correct regardless of how `gram_hyp_paper` turns out.
- Their code is not released and the method admits two readings of the hyperbolic branch. v1
  and v2 implement both; v3 is v2 at the right recipe.

## What would change this

Train `gram_hyp_paper` and evaluate it on all five benchmarks:

    sbatch --array=41 slurm_scripts/b_grid_pretrain.sh          # train
    SCA_REPRO_HYP_ARM=gram_hyp_paper sbatch --array=5-9 slurm_scripts/repro_baselines_eval.sh

Then, and only then, is there a reproduction number worth discussing. If it lands near 56.6 the
same-environment table gains a real HyperGRAM row. If it collapses again at the *correct*
recipe, that is a finding about reproducibility that can be stated honestly — but it cannot be
stated from `gram_hyp2`.

## Record of the error

The 37.4 figure was quoted repeatedly as "our HyperGRAM reproduction does not work", including
in commit messages and in `slurm_scripts/repro_baselines_eval.sh`. Every one of those citations
rests on a run at the wrong learning rate. The launcher comment has been corrected. Anything
written before 2026-08-23 that cites 37.4 or 51.0 as HyperGRAM's reproduced performance is
wrong for this reason.
