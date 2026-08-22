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
#SBATCH --job-name=e1_fullft
#SBATCH -o ./slurm_scripts/logs/e1_fullft_%j.out
#SBATCH -e ./slurm_scripts/logs/e1_fullft_%j.out
# The EXISTING SCA full-finetuning checkpoints on the four transfer benchmarks.
#
# Under full finetuning nothing is frozen, so the ITM cross-encoder trains exactly as
# GRAM's does -- which makes these checkpoints a free test of the diagnosis that SCA loses
# 5-6 points at the reranking stage because LoRA freezes multimodal_encoder. Both arms were
# already trained (A6_full_ft at 2e-5 -> 53.4 on MSR-VTT, T4_fullft_lr1e4 at 1e-4 -> 53.0,
# unstable) and only ever evaluated on MSR-VTT, so the transfer answer costs an eval rather
# than two pretrains.
#
# What to look for: if full finetuning closes the ITM gap on DiDeMo/ActivityNet even while
# its MSR-VTT number stays below the LoRA arm's, the reranker is confirmed as the
# bottleneck and X1 (cross-encoder trainable, encoders still adapted) is the configuration
# that should get both effects at once.
#
#   SCA_FULLFT_2E5=workdir_pretrain/a6_fullft \
#   SCA_FULLFT_1E4=workdir_pretrain/t4_fullft1e4 \
#     sbatch slurm_scripts/e1_zs_fullft.sh
#
# Per-cell config-aware done markers: resubmit to resume.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
# Resolve from CODE_DIR, which every launcher has already cd'd into. Under sbatch, $0 is a
# COPY of the script in Slurm's spool directory (/var/spool/slurmd/job<N>/slurm_script), so
# "$(dirname "$0")/.." points at /var/spool/slurmd and there is no scripts/ there. This source
# has been failing in every Slurm job since it was written. Nothing surfaced because the
# callers use `cell_is_done ... && continue`: an undefined function returns 127, the && short-
# circuits, and the cell simply runs. So the resume-skip and the config fingerprinting have
# both been inert under Slurm, which is also why eval cells show up twice in the logs.
HELPER="${CODE_DIR:-.}/scripts/cell_done.sh"
[ -f "$HELPER" ] || HELPER="$(dirname "$0")/../scripts/cell_done.sh"
# shellcheck source=/dev/null
source "$HELPER" || { echo "FATAL: cannot source $HELPER" >&2; exit 2; }
command -v cell_is_done >/dev/null || {
  echo "FATAL: sourced $HELPER but cell_is_done is not defined." >&2; exit 2; }
mkdir -p slurm_scripts/logs

best_ckpt() {
  local b; b=$(ls -t "$1"/ckpt/best_*.pt 2>/dev/null | head -1)
  [ -n "$b" ] || b=$(ls "$1"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1)
  echo "$b"
}

declare -A ARMS=( [fullft2e5]="${SCA_FULLFT_2E5:-workdir_pretrain/a6_fullft}"
                  [fullft1e4]="${SCA_FULLFT_1E4:-workdir_pretrain/t4_fullft1e4}" )
for a in "${!ARMS[@]}"; do
  d="${ARMS[$a]}"
  if [ ! -d "$d" ]; then
    echo "FATAL: $a workdir '$d' not found. Point SCA_FULLFT_2E5 / SCA_FULLFT_1E4 at the" >&2
    echo "       full-finetuning pretrain directories (ls workdir_pretrain to find them)." >&2
    exit 2
  fi
  c=$(best_ckpt "$d")
  [ -n "$c" ] && [ -f "$c" ] || { echo "FATAL: no checkpoint under $d/ckpt" >&2; exit 2; }
  echo "$a ckpt: $c"
done

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0
for arm in fullft2e5 fullft1e4; do
  ckpt=$(best_ckpt "${ARMS[$arm]}")
  for bench in didemo activitynet vatex audiocaps; do
    cfg="benchmark_eval/configs_e1/sca_${bench}.json"
    out="workdir/e1_zs/${arm}_${bench}"
    cell_is_done "$out" "$cfg" && { echo "== [$arm/$bench] already done, skip"; continue; }
    mkdir -p "$out"
    echo "== [$arm/$bench] START $(date +%T)"
    EVAL_CKPT="$ckpt" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
      --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
      --master_port $((9600 + ${SLURM_JOB_ID:-$$} % 200)) \
      ./benchmark_eval/run_eval.py --config "$cfg" --output_dir "$out" 2>&1 \
      | { grep -v --line-buffered -E "$NOISE" || true; }
    rc=$?
    if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$arm/$bench] OK $(date +%T)"
    else echo "== [$arm/$bench] FAILED rc=$rc" >&2; rc_all=$rc; fi
  done
done
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
