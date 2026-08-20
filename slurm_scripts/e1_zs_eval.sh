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
#SBATCH --job-name=e1_zs
#SBATCH -o ./slurm_scripts/logs/e1_zs_%j.out
#SBATCH -e ./slurm_scripts/logs/e1_zs_%j.out
# E1 zero-shot grid: SCA + GRAM-repro pretrain checkpoints on DiDeMo / ActivityNet /
# VATEX(431-clip audio subset) / AudioCaps, ITM protocol (bidirectional, rerank 50).
# MSR-VTT zs is already measured during pretraining. Cells that finished are skipped on
# resubmission (done-marker per cell), so rerun the same sbatch after a timeout.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
source "$(dirname "$0")/../scripts/cell_done.sh"
mkdir -p slurm_scripts/logs

best_ckpt() {
  local b
  b=$(ls -t "$1"/ckpt/best_*.pt 2>/dev/null | head -1)
  [ -n "$b" ] || b=$(ls "$1"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1)
  echo "$b"
}
SCA_CKPT="${SCA_ZS_CKPT:-$(best_ckpt workdir_pretrain/sca)}"
GRAM_CKPT="${GRAM_ZS_CKPT:-$(best_ckpt workdir_pretrain/gram)}"
[ -f "$SCA_CKPT" ]  || { echo "FATAL: no SCA pretrain ckpt (workdir_pretrain/sca)" >&2; exit 1; }
[ -f "$GRAM_CKPT" ] || { echo "FATAL: no GRAM pretrain ckpt (workdir_pretrain/gram)" >&2; exit 1; }
echo "SCA ckpt:  $SCA_CKPT"
echo "GRAM ckpt: $GRAM_CKPT"

rc_all=0
for cell in sca_didemo sca_activitynet sca_vatex sca_audiocaps \
            gram_didemo gram_activitynet gram_vatex gram_audiocaps; do
  fam="${cell%%_*}"
  out="workdir/e1_zs/${cell}${E1_TAG:-}"
  cfg="benchmark_eval/configs_e1/$cell.json"
  if cell_is_done "$out" "$cfg"; then echo "== [$cell] already done, skip"; continue; fi
  ckpt="$SCA_CKPT"; [ "$fam" = gram ] && ckpt="$GRAM_CKPT"
  echo "== [$cell] START $(date +%T)  ckpt=$ckpt"
  mkdir -p "$out"
  EVAL_CKPT="$ckpt" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
    --nproc_per_node 4 --master_port $((9200 + RANDOM % 500)) \
    ./benchmark_eval/run_eval.py --config "benchmark_eval/configs_e1/$cell.json" \
    --output_dir "$out" 2>&1 \
    | { grep -v --line-buffered -E "mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($" || true; }
  rc=$?
  if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$cell] OK $(date +%T)"
  else echo "== [$cell] FAILED rc=$rc" >&2; rc_all=$rc; fi
done
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
