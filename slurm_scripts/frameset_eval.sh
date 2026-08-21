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
#SBATCH --job-name=fs_eval
#SBATCH -o ./slurm_scripts/logs/fs_eval_%j.out
#SBATCH -e ./slurm_scripts/logs/fs_eval_%j.out
# Evaluate frame-set arms on all five benchmarks, scored WITH the frame set.
#
# These arms must not go through benchmark_eval/configs_e1: those build one slot per
# modality from the pooled feat_v, so a frame-set checkpoint would be scored as a model that
# was never trained. configs_frames/ sets sca_frame_slots, sca_query_weighting, sca_tau_w and
# dump_frame_feats, and evaluation_mm raises rather than falling back if the flags are on but
# no per-frame features arrive.
#
# The FINAL checkpoint, matching e1_final_ckpt.sh -- never best_*.pt, which save_best selects
# on the aggregator score rather than the reported metric.
#
#   sbatch --dependency=afterok:<t6_jobid> slurm_scripts/frameset_eval.sh
#   SCA_FS_ARMS="t6_frameset t7_frameset_4f" sbatch slurm_scripts/frameset_eval.sh
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
source "$(dirname "$0")/../scripts/cell_done.sh"
mkdir -p slurm_scripts/logs

final_ckpt() { ls "$1"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1; }

ARMS="${SCA_FS_ARMS:-t6_frameset t7_frameset_4f t8_frameset_tau005}"
FOUND=""
for arm in $ARMS; do
  c=$(final_ckpt "workdir_pretrain/$arm")
  if [ -n "$c" ] && [ -f "$c" ]; then echo "$arm -> $c"; FOUND="$FOUND $arm"
  else echo "SKIP $arm: no model_step_*.pt yet (still training or never ran)"; fi
done
[ -n "$FOUND" ] || { echo "FATAL: no frame-set arm has a final checkpoint" >&2; exit 2; }

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0
for arm in $FOUND; do
  ckpt=$(final_ckpt "workdir_pretrain/$arm")
  for bench in msrvtt didemo activitynet vatex audiocaps; do
    cfg="benchmark_eval/configs_frames/sca_${bench}.json"
    [ -f "$cfg" ] || { echo "== [$arm/$bench] SKIP: no $cfg" >&2; rc_all=2; continue; }
    out="workdir/e1_frames/${arm}_${bench}"
    cell_is_done "$out" "$cfg" && { echo "== [$arm/$bench] already done, skip"; continue; }
    mkdir -p "$out"
    echo "== [$arm/$bench] START $(date +%T)"
    EVAL_CKPT="$ckpt" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
      --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
      --master_port $((9100 + ${SLURM_JOB_ID:-$$} % 90)) \
      ./benchmark_eval/run_eval.py --config "$cfg" --output_dir "$out" 2>&1 \
      | { grep -v --line-buffered -E "$NOISE" || true; }
    rc=$?
    if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$arm/$bench] OK $(date +%T)"
    else echo "== [$arm/$bench] FAILED rc=$rc" >&2; rc_all=$rc; fi
  done
done
echo
echo "  python3 scripts/raw_vs_itm.py --root workdir/e1_frames --pivot"
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
