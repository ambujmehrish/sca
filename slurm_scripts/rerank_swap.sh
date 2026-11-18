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
#SBATCH --job-name=rrswap
#SBATCH --array=0-2
#SBATCH -o ./slurm_scripts/logs/rrswap_%x_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/rrswap_%x_%A_%a.out
# Reranker swap at 8 frames: which stage loses SCALAR its margin, the candidates or the reranker?
#
#   PHASE=dump  each model scores its own candidates and dumps its stage-1 matrix
#               0 sca (T9 final)   1 gram   2 vast (the common initialization, GRAM class)
#   PHASE=swap  a model's reranker (its encoders + cross-encoder + ITM head) reranks another
#               model's top-50; its own-candidate result is recomputed in the same run as a control
#               0 gram reranks sca's candidates
#               1 sca  reranks gram's candidates
#               2 vast reranks sca's candidates
#
#   export GRAM_RELEASED_CKPT=/path/to/GRAM/ckpt/model_step_459.pt
#   jid=$(sbatch --parsable --export=ALL,PHASE=dump slurm_scripts/rerank_swap.sh)
#   sbatch --dependency=afterok:${jid} --export=ALL,PHASE=swap slurm_scripts/rerank_swap.sh
#
# Both phases must run on the same GPU count: the swap requires the dump's query and gallery
# order to match its own exactly, and raises otherwise.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
# shellcheck source=/dev/null
source "$CODE_DIR/scripts/cell_done.sh" || { echo "FATAL: cannot source scripts/cell_done.sh" >&2; exit 2; }
command -v cell_is_done >/dev/null || {
  echo "FATAL: sourced scripts/cell_done.sh but cell_is_done is not defined." >&2; exit 2; }
# shellcheck source=/dev/null
source "$CODE_DIR/scripts/parity_common.sh" || exit 2
mkdir -p slurm_scripts/logs

IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
ROOT_OUT="workdir/rerank_swap"
case "${PHASE:-}" in
  dump) PAIRS=("sca:sca" "gram:gram" "vast:vast") ;;
  swap) PAIRS=("gram:sca" "sca:gram" "vast:sca") ;;
  *) echo "FATAL: set PHASE=dump or PHASE=swap" >&2; exit 2 ;;
esac
PAIR="${PAIRS[$IDX]:-}"
[ -n "$PAIR" ] || { echo "FATAL: array index '$IDX' out of range 0-2" >&2; exit 2; }
RERANKER="${PAIR%%:*}"; CANDS="${PAIR##*:}"
check_ckpt "$RERANKER" || exit 2
CKPT=$(ckpt_for "$RERANKER")
echo "PHASE=$PHASE  reranker=$RERANKER  candidates=$CANDS"

rc_all=0
for bench in msrvtt didemo activitynet vatex audiocaps; do
  cfg=$(cfg_for "$RERANKER" "$bench" fs8)
  if [ "$PHASE" = dump ]; then
    run_cell "$ROOT_OUT/own_${RERANKER}_${bench}" "$cfg" "$CKPT" \
      SCA_DUMP_RERANK="$CODE_DIR/$ROOT_OUT/dumps/${RERANKER}_${bench}" || rc_all=1
    n=$(ls "$ROOT_OUT/dumps/${RERANKER}_${bench}"/rerank_*.pt 2>/dev/null | wc -l)
    [ "$n" -eq 1 ] || { echo "FATAL: expected one dump for ${RERANKER}_${bench}, found $n" >&2; rc_all=1; }
  else
    dumps=( "$ROOT_OUT/dumps/${CANDS}_${bench}"/rerank_*.pt )
    if [ "${#dumps[@]}" -ne 1 ] || [ ! -f "${dumps[0]}" ]; then
      echo "FATAL: need exactly one dump in $ROOT_OUT/dumps/${CANDS}_${bench} (run PHASE=dump first)" >&2
      rc_all=1; continue
    fi
    run_cell "$ROOT_OUT/swap_${RERANKER}_reranks_${CANDS}_${bench}" "$cfg" "$CKPT" \
      SCA_EXTERNAL_CANDIDATES="$CODE_DIR/${dumps[0]}" || rc_all=1
  fi
done
echo "  python3 scripts/summarize_parity.py"
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
