#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=subset
#SBATCH --array=0-3
#SBATCH -o ./slurm_scripts/logs/subset_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/subset_%A_%a.out
# Modality-subset ladder (APPENDIX): {V} and {V,A} rungs of the tvas benchmarks, SCA (T9)
# and the released GRAM checkpoint. The {V,A,S} rung is the canonical, already-measured
# cell -- never re-run, so the ladder's endpoint IS the main table by construction.
#
# One array task per (model, bench); each runs the tv and tva rungs sequentially.
#
#   GRAM_RELEASED_CKPT=/path/to/released.pt sbatch slurm_scripts/subset_eval.sh
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" \
  || { echo "FATAL: $MODELS_DIR/env.sh not found" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTHONUNBUFFERED=1
HELPER="${CODE_DIR:-.}/scripts/cell_done.sh"
[ -f "$HELPER" ] || HELPER="$(dirname "$0")/../scripts/cell_done.sh"
# shellcheck source=/dev/null
source "$HELPER" || { echo "FATAL: cannot source $HELPER" >&2; exit 2; }
command -v cell_is_done >/dev/null || {
  echo "FATAL: sourced $HELPER but cell_is_done is not defined." >&2; exit 2; }
mkdir -p slurm_scripts/logs

CELLS=(sca:msrvtt sca:vatex gram:msrvtt gram:vatex)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[ -n "$IDX" ] || { echo "FATAL: no array index. sbatch this, or pass 0-3." >&2; exit 2; }
CELL="${CELLS[$IDX]:-}"
[ -n "$CELL" ] || { echo "FATAL: index $IDX out of range (0-3)" >&2; exit 2; }
MODEL="${CELL%%:*}"; BENCH="${CELL##*:}"

if [ "$MODEL" = gram ]; then
  if [ -z "${GRAM_RELEASED_CKPT:-}" ]; then
    echo "FATAL: set GRAM_RELEASED_CKPT -- same file as the GRAM* rows of Tables 1/2." >&2
    exit 2
  fi
  CKPT="$GRAM_RELEASED_CKPT"
else
  CKPT="workdir_pretrain/t9_qweight_only/ckpt/model_step_5330.pt"
fi
[ -f "$CKPT" ] || { echo "FATAL: checkpoint not at $CKPT" >&2; exit 2; }

python3 scripts/make_subset_configs.py || exit 2

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0
for SUB in tv tva; do
  CFG="benchmark_eval/configs_subsets/${SUB}/${MODEL}_${BENCH}.json"
  OUT="workdir/e1_subsets/${MODEL}_${BENCH}_${SUB}"
  cell_is_done "$OUT" "$CFG" && { echo "== [$MODEL/$BENCH/$SUB] already done, skip"; continue; }
  mkdir -p "$OUT"
  echo "== [$MODEL/$BENCH/$SUB] START $(date +%T)  ckpt=$CKPT"
  EVAL_CKPT="$CKPT" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
    --nproc_per_node 4 --master_port $((9950 + IDX)) \
    ./benchmark_eval/run_eval.py --config "$CFG" --output_dir "$OUT" 2>&1 \
    | tee "$OUT/eval.log" | { grep -v --line-buffered -E "$NOISE" || true; }
  rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ] && [ "$MODEL" = gram ]; then
    # contra_head_d: the unused depth head, absent from the released checkpoint by
    # construction (see missing_eval.sh); every other key still gates the cell.
    python3 scripts/verify_ckpt_load.py "$OUT/eval.log" --allow-prefix contra_head_d || {
      echo "== [$MODEL/$BENCH/$SUB] LOAD NOT VERIFIED -- cell refused" >&2; rc=3; }
  fi
  if [ $rc -eq 0 ]; then cell_mark_done "$OUT" "$CFG"; echo "== [$MODEL/$BENCH/$SUB] OK $(date +%T)"
  else echo "== [$MODEL/$BENCH/$SUB] FAILED rc=$rc" >&2; rc_all=$rc; fi
done
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
