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
#SBATCH --job-name=e1_released
#SBATCH -o ./slurm_scripts/logs/e1_released_%j.out
#SBATCH -e ./slurm_scripts/logs/e1_released_%j.out
# GRAM's OFFICIALLY RELEASED checkpoint on the four transfer benchmarks.
#
# Why this matters more than it looks: the released weights were trained by the authors at
# their own recipe (lr 1e-4, batch 128), so this row is "GRAM at its own recipe" WITHOUT
# training anything. Table 1 already has it on MSR-VTT (52.5); the transfer benchmarks were
# only ever run on our lr-2e-5 reproduction, which is why Table 2's GRAM row is off-recipe.
#
# It also settles the open question in Table 2: published GRAM reports 59.0 on ActivityNet
# against our reproduction's 52.0, a 7-point gap far beyond the ~2-point environment offset.
# If the released checkpoint scores near 59 here, the gap is the RECIPE and Table 2's SCA
# lead on ActivityNet is in danger; if it scores near 52, the gap is something else and the
# lead stands. Either way we learn it from an eval, days before gram_paper finishes.
#
#   GRAM_RELEASED_CKPT=/path/to/released.pt sbatch slurm_scripts/e1_zs_released.sh
#
# Per-cell done markers: resubmit to resume.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
source "$(dirname "$0")/../scripts/cell_done.sh"
mkdir -p slurm_scripts/logs

# No default guess here: the released checkpoint is not in this repo and silently falling
# back to workdir_pretrain/gram would relabel our 2e-5 reproduction as the released model --
# exactly the kind of mix-up the provenance work is meant to prevent.
if [ -z "${GRAM_RELEASED_CKPT:-}" ]; then
  echo "FATAL: set GRAM_RELEASED_CKPT to the released GRAM checkpoint -- the same file" >&2
  echo "       used for the MSR-VTT GRAM* row in Table 1." >&2
  exit 2
fi
CKPT="$GRAM_RELEASED_CKPT"
[ -f "$CKPT" ] || { echo "FATAL: GRAM_RELEASED_CKPT=$CKPT does not exist" >&2; exit 1; }
echo "released GRAM ckpt: $CKPT"

rc_all=0
for bench in didemo activitynet vatex audiocaps; do
  cell="gram_${bench}"
  out="workdir/e1_zs/released_${bench}"
  cfg="benchmark_eval/configs_e1/${cell}.json"
  cell_is_done "$out" "$cfg" && { echo "== [$cell] already done, skip"; continue; }
  mkdir -p "$out"
  echo "== [released/$bench] START $(date +%T)"
  EVAL_CKPT="$CKPT" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
    --nproc_per_node 4 --master_port $((9300 + ${SLURM_JOB_ID:-$$} % 200)) \
    ./benchmark_eval/run_eval.py --config "benchmark_eval/configs_e1/${cell}.json" \
    --output_dir "$out" 2>&1 \
    | { grep -v --line-buffered -E "mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($" || true; }
  rc=$?
  if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [released/$bench] OK $(date +%T)"
  else echo "== [released/$bench] FAILED rc=$rc" >&2; rc_all=$rc; fi
done
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
