#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=released_grids
#SBATCH -o ./slurm_scripts/logs/released_grids_%j.out
#SBATCH -e ./slurm_scripts/logs/released_grids_%j.out
# The released GRAM checkpoint through EVERY remaining grid, not just zero-shot retrieval.
#
# Baselines are frozen: GRAM is now reported from its released weights plus its published
# numbers, and PMRL/HyperGRAM from published numbers alone. But published numbers exist
# only for plain retrieval -- nobody reports GRAM at 50% missing audio, or its score
# calibration, or its behaviour across modality cardinalities. For Tables 4, 5 and 6 the
# ONLY way to keep a same-environment GRAM baseline is to run the released checkpoint
# through those grids ourselves. That is what this does.
#
# Cells:
#   (1) E4-ITM missing modality, MSR-VTT, drop rates 50/90 -- the reported metric
#   (2) E4-ITM missing modality on the three transfer benchmarks
#   (3) feature dumps per benchmark, which feed the E5 cardinality / E6 calibration
#       analyses and the E8 diagnostics without a second forward pass
#
#   GRAM_RELEASED_CKPT=/path/to/released.pt sbatch slurm_scripts/released_all_grids.sh
#
# Per-cell done markers: resubmit to resume. Results land under workdir/*_released/ and
# results/e4_released/, kept apart from the reproduction's directories so the two GRAM
# generations can never be confused when extracting.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
mkdir -p slurm_scripts/logs results/e4_released/feats

if [ -z "${GRAM_RELEASED_CKPT:-}" ]; then
  echo "FATAL: set GRAM_RELEASED_CKPT to the released GRAM checkpoint -- the same file" >&2
  echo "       used for the MSR-VTT GRAM* row in Table 1." >&2
  exit 2
fi
CKPT="$GRAM_RELEASED_CKPT"
[ -f "$CKPT" ] || { echo "FATAL: GRAM_RELEASED_CKPT=$CKPT does not exist" >&2; exit 1; }
echo "released GRAM ckpt: $CKPT"

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0

# ---- (1)+(2) E4-ITM: the metric the missing-modality tables actually report --------------
# Reuses the existing gram_* cells; only the checkpoint and the output directory change.
for cell in gram_r50 gram_r90 \
            gram_didemo_r50 gram_didemo_r90 \
            gram_activitynet_r50 gram_activitynet_r90 \
            gram_audiocaps_r50 gram_audiocaps_r90; do
  cfg="benchmark_eval/configs_e4itm/${cell}.json"
  [ -f "$cfg" ] || { echo "== [$cell] config missing, skip" >&2; continue; }
  out="workdir/e4_itm_released/${cell}"
  [ -f "$out/.done" ] && { echo "== [$cell] already done, skip"; continue; }
  mkdir -p "$out"
  echo "== [$cell] START $(date +%T)"
  EVAL_CKPT="$CKPT" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
    --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
    --master_port $((9400 + ${SLURM_JOB_ID:-$$} % 200)) \
    ./benchmark_eval/run_eval.py --config "$cfg" --output_dir "$out" 2>&1 \
    | { grep -v --line-buffered -E "$NOISE" || true; }
  rc=$?
  if [ $rc -eq 0 ]; then touch "$out/.done"; echo "== [$cell] OK $(date +%T)"
  else echo "== [$cell] FAILED rc=$rc" >&2; rc_all=$rc; fi
done

# ---- (3) feature dumps: one forward pass each, reused by E5 / E6 / E8 --------------------
for bench in msrvtt didemo activitynet audiocaps; do
  feats="results/e4_released/feats/gram_${bench}.pt"
  [ -f "$feats" ] && { echo "== [feats/$bench] already dumped, skip"; continue; }
  cfg="benchmark_eval/configs_e1/gram_${bench}.json"
  # No fallback: the e4itm cells carry eval_mask_rate, so falling back to one would dump
  # features with modalities already dropped and every downstream analysis would silently
  # be computed on masked features.
  [ -f "$cfg" ] || { echo "FATAL: $cfg missing -- refusing to dump features from a masked config" >&2; rc_all=1; continue; }
  echo "== [feats/$bench] START $(date +%T)  cfg=$cfg"
  EVAL_CKPT="$CKPT" python3 evaluation/run_eval_grids.py --config "$cfg" \
    --dump_features "$feats" 2>&1 | { grep -v --line-buffered -E "$NOISE" || true; }
  rc=$?
  [ $rc -eq 0 ] || { echo "== [feats/$bench] FAILED rc=$rc" >&2; rc_all=$rc; continue; }
  for s in 0 1 2; do
    python3 evaluation/run_eval_grids.py --features "$feats" --seed $s \
      --rates 0 0.25 0.5 0.75 0.9 \
      --out "results/e4_released/gram_${bench}_s${s}.json" \
      || { echo "== [feats/$bench/s$s] GRID FAILED" >&2; rc_all=1; }
  done
  echo "== [feats/$bench] OK $(date +%T)"
done

echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
