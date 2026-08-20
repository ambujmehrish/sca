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
#SBATCH --job-name=e4_itm
#SBATCH -o ./slurm_scripts/logs/e4_itm_%j.out
#SBATCH -e ./slurm_scripts/logs/e4_itm_%j.out
# E4 confirmed on the TABLE metric: full ITM-reranked eval with modalities dropped at the
# ENCODER-OUTPUT level (both stages see the missing modality). Answers the reviewer
# question "your robustness claim is on a metric you don't report" -- 3 arms x 2 rates
# (rate 0 is the existing zero-shot row). Per-cell done markers: resubmit to resume.
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
declare -A CKPT=( [sca]="$(best_ckpt workdir_pretrain/t1_lr1e4)" \
                  [sca_base]="$(best_ckpt workdir_pretrain/sca)" \
                  [gram]="$(best_ckpt workdir_pretrain/gram)" \
                  [gram_lora]="$(best_ckpt workdir_pretrain/gram_lora)" )
for a in "${!CKPT[@]}"; do [ -f "${CKPT[$a]}" ] || { echo "FATAL: $a ckpt missing" >&2; exit 1; }; done

rc_all=0
for arm in sca sca_base gram gram_lora; do
  for rate in 50 90; do
    cell="${arm}_r${rate}"; out="workdir/e4_itm/$cell"
    cfg="benchmark_eval/configs_e4itm/${cell}.json"
    cell_is_done "$out" "$cfg" && { echo "== [$cell] already done, skip"; continue; }
    mkdir -p "$out"
    echo "== [$cell] START $(date +%T)  ckpt=${CKPT[$arm]}"
    EVAL_CKPT="${CKPT[$arm]}" srun python3 -m torch.distributed.launch --nnodes 1 \
      --node_rank 0 --nproc_per_node 4 --master_port $((9700 + RANDOM % 200)) \
      ./benchmark_eval/run_eval.py --config "benchmark_eval/configs_e4itm/${cell}.json" \
      --output_dir "$out" 2>&1 \
      | { grep -v --line-buffered -E "mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($" || true; }
    rc=$?
    if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$cell] OK $(date +%T)"
    else echo "== [$cell] FAILED rc=$rc" >&2; rc_all=$rc; fi
  done
done
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
