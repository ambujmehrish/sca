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
#SBATCH --job-name=b_grid_eval
#SBATCH -o ./slurm_scripts/logs/b_grid_eval_%j.out
#SBATCH -e ./slurm_scripts/logs/b_grid_eval_%j.out
# The four batch-size x LoRA-capacity arms on all five benchmarks.
#
# B2 and B4 are rank 32 with alpha 64. Before utils/lora_geometry.py they would have loaded
# CLEANLY into the eval configs' rank-8/alpha-16 defaults on the alpha axis and run at a
# quarter of the adapter strength they were trained with, reporting plausible numbers from a
# model that never existed. run_eval.py now takes the geometry from the checkpoint, and the
# log line '[LoRA] geometry taken from the checkpoint' is the receipt -- if it is absent for
# the r32 arms, stop and do not trust the numbers.
#
#   sbatch slurm_scripts/b_grid_eval.sh
#   SCA_B_ARMS="b1_bs128_r8" sbatch slurm_scripts/b_grid_eval.sh    # one arm
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
source "$(dirname "$0")/../scripts/cell_done.sh"
mkdir -p slurm_scripts/logs

best_ckpt() {
  local b; b=$(ls -t "$1"/ckpt/best_*.pt 2>/dev/null | head -1)
  [ -n "$b" ] || b=$(ls "$1"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1)
  echo "$b"
}

ARMS="${SCA_B_ARMS:-b1_bs128_r8 b2_bs128_r32 b3_bs512_r8 b4_bs512_r32 b5_bs128_xenc b6_bs512_xenc}"
for arm in $ARMS; do
  d="workdir_pretrain/$arm"
  [ -d "$d" ] || { echo "FATAL: $d not found -- did the pretrain run?" >&2; exit 2; }
  c=$(best_ckpt "$d"); [ -n "$c" ] && [ -f "$c" ] \
    || { echo "FATAL: no checkpoint under $d/ckpt -- the pretrain did not get far enough" >&2; exit 2; }
  echo "$arm -> $c"
done

cfg_for() { case "$1" in msrvtt) echo "benchmark_eval/configs_depth/sca_msrvtt_tvas.json" ;;
                         *) echo "benchmark_eval/configs_e1/sca_$1.json" ;; esac; }
NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0
for arm in $ARMS; do
  ckpt=$(best_ckpt "workdir_pretrain/$arm")
  for bench in msrvtt didemo activitynet vatex audiocaps; do
    cfg=$(cfg_for "$bench")
    out="workdir/e1_zs/${arm}_${bench}"
    cell_is_done "$out" "$cfg" && { echo "== [$arm/$bench] already done, skip"; continue; }
    mkdir -p "$out"
    echo "== [$arm/$bench] START $(date +%T)"
    EVAL_CKPT="$ckpt" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
      --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
      --master_port $((9400 + ${SLURM_JOB_ID:-$$} % 150)) \
      ./benchmark_eval/run_eval.py --config "$cfg" --output_dir "$out" 2>&1 \
      | { grep -v --line-buffered -E "$NOISE" || true; }
    rc=$?
    if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$arm/$bench] OK $(date +%T)"
    else echo "== [$arm/$bench] FAILED rc=$rc" >&2; rc_all=$rc; fi
  done
done
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
