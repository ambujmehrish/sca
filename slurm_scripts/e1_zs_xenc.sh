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
#SBATCH --job-name=e1_xenc
#SBATCH -o ./slurm_scripts/logs/e1_xenc_%j.out
#SBATCH -e ./slurm_scripts/logs/e1_xenc_%j.out
# The four ITM cross-encoder arms on the four transfer benchmarks -- the missing cells.
#
# On MSR-VTT these arms revealed something the LoRA arms never showed: x1_xenc_full at
# lr 2e-5 reaches V2T 50.6 against SCA's 49.2, and is the only arm so far positive against
# the released GRAM checkpoint in BOTH directions (+1.2 T2V, +0.1 V2T). Every other arm wins
# text-to-video and loses video-to-text, which is the "V2T is parity, not a win" caveat the
# paper has carried from the start.
#
# Whether that holds away from the selection benchmark is exactly what is unknown. It is
# also the deciding evidence for the single-configuration choice: on the transfer benchmarks
# SCA at lr 2e-5 already beats SCA at 1e-4 on 8 of 10 cells, so if the cross-encoder arm
# keeps its V2T gain off MSR-VTT it becomes the configuration to report everywhere.
#
# Eval only -- no training. 16 cells, config-aware done markers, resubmit to resume.
#
#   sbatch slurm_scripts/e1_zs_xenc.sh
#   SCA_XENC_ARMS="x1_xenc_full_lr2e5" sbatch slurm_scripts/e1_zs_xenc.sh   # subset
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

ARMS="${SCA_XENC_ARMS:-x1_xenc_full x1_xenc_full_lr2e5 x2_xenc_r64 x2_xenc_r64_lr2e5}"
for arm in $ARMS; do
  d="workdir_pretrain/$arm"
  [ -d "$d" ] || { echo "FATAL: $d not found" >&2; exit 2; }
  c=$(best_ckpt "$d")
  [ -n "$c" ] && [ -f "$c" ] || { echo "FATAL: no checkpoint under $d/ckpt" >&2; exit 2; }
  echo "$arm -> $c"
done

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0
for arm in $ARMS; do
  ckpt=$(best_ckpt "workdir_pretrain/$arm")
  for bench in didemo activitynet vatex audiocaps; do
    # the sca_* eval configs carry the centroid scorer and the corrected max_caption_len;
    # only the checkpoint changes between arms
    cfg="benchmark_eval/configs_e1/sca_${bench}.json"
    out="workdir/e1_zs/${arm}_${bench}"
    cell_is_done "$out" "$cfg" && { echo "== [$arm/$bench] already done, skip"; continue; }
    mkdir -p "$out"
    echo "== [$arm/$bench] START $(date +%T)"
    EVAL_CKPT="$ckpt" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
      --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
      --master_port $((9800 + ${SLURM_JOB_ID:-$$} % 150)) \
      ./benchmark_eval/run_eval.py --config "$cfg" --output_dir "$out" 2>&1 \
      | { grep -v --line-buffered -E "$NOISE" || true; }
    rc=$?
    if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$arm/$bench] OK $(date +%T)"
    else echo "== [$arm/$bench] FAILED rc=$rc" >&2; rc_all=$rc; fi
  done
done
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
