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
#SBATCH --job-name=repro
#SBATCH --array=0-14
#SBATCH -o ./slurm_scripts/logs/repro_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/repro_%A_%a.out
# RETIRED -- do not run. Superseded by slurm_scripts/hypergram_authors.sh.
#
# This launcher evaluated OUR reimplementations of the competing aggregators. Two of those
# reimplementations are now known not to match the released code:
#
#   hypergram  differs from github.com/uta-smile/HyperGram in six substantive ways -- no
#              learnable curvature, no curvature learning-rate group, and no scale matching
#              between the Euclidean and hyperbolic volumes before they are mixed
#              (experiments/results/HYPERGRAM_STATUS.md)
#   pmrl       unverified in exactly the same way; their repo implements PMRL too
#              (geometry_mode pmrl / pmrl_volume / hybrid_pmrl) and ours was never checked
#              against it
#
# Their code is public and is the same VAST/GRAM fork we build on, so every baseline now comes
# from the authors' implementation rather than ours. Run:
#
#   sbatch slurm_scripts/hypergram_authors.sh
#
# Kept on disk as the record of what was run, and because the gram_lora appendix control is
# still ours by construction -- if that control is ever wanted, run --array=10-14 knowingly.
# Nothing from the pmrl or hypergram indices belongs in a table.
#
set -uo pipefail

# Refuse the superseded rows outright. Keeping the script runnable for gram_lora while it
# still answers to `sbatch` with no arguments is how a retired baseline quietly reappears in a
# table months later.
case "${SLURM_ARRAY_TASK_ID:-${1:-}}" in
  0|1|2|3|4|5|6|7|8|9)
    echo "FATAL: indices 0-9 are RETIRED (pmrl, hypergram from OUR reimplementation)." >&2
    echo "       Those do not match the released code. Use the authors' implementation:" >&2
    echo "         sbatch slurm_scripts/hypergram_authors.sh" >&2
    echo "       See experiments/results/HYPERGRAM_STATUS.md." >&2
    exit 2 ;;
esac

source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" \
  || { echo "FATAL: $MODELS_DIR/env.sh not found -- run scripts/prefetch_models.py first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
HELPER="${CODE_DIR:-.}/scripts/cell_done.sh"
[ -f "$HELPER" ] || HELPER="$(dirname "$0")/../scripts/cell_done.sh"
# shellcheck source=/dev/null
source "$HELPER" || { echo "FATAL: cannot source $HELPER" >&2; exit 2; }
command -v cell_is_done >/dev/null || {
  echo "FATAL: sourced $HELPER but cell_is_done is not defined." >&2; exit 2; }
mkdir -p slurm_scripts/logs

METHODS=(pmrl hypergram gram_lora)
BENCHES=(msrvtt didemo activitynet vatex audiocaps)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[ -n "$IDX" ] || { echo "FATAL: no array index. sbatch this, or pass 0-14 for one cell." >&2; exit 2; }
METHOD="${METHODS[$((IDX / 5))]:-}"
BENCH="${BENCHES[$((IDX % 5))]:-}"
[ -n "$METHOD" ] && [ -n "$BENCH" ] || { echo "FATAL: index $IDX out of range (0-14)" >&2; exit 2; }

# the trained arm behind each row. gram_hyp2 is the v2 reading; gram_hyp is v1.
case "$METHOD" in
  pmrl)      ARM="${SCA_REPRO_PMRL_ARM:-pmrl_lora}" ;;
  hypergram) ARM="${SCA_REPRO_HYP_ARM:-gram_hyp2}" ;;
  gram_lora) ARM="${SCA_REPRO_GRAM_ARM:-gram_lora}" ;;
esac
CKPT=$(ls "workdir_pretrain/$ARM"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1)
[ -n "$CKPT" ] && [ -f "$CKPT" ] || {
  echo "FATAL: no model_step_*.pt under workdir_pretrain/$ARM -- ls workdir_pretrain to check" >&2
  exit 2; }

CFG="benchmark_eval/configs_repro/${METHOD}_${BENCH}.json"
[ -f "$CFG" ] || { echo "FATAL: $CFG not found" >&2; exit 2; }
# The eval config's model_type must match what the checkpoint actually is. Scoring a PMRL
# checkpoint through the volume, or a hypergraph checkpoint through the centroid, produces a
# complete set of plausible numbers for a model that was never trained -- no shape mismatch,
# no error, and a table that is wrong in a way nobody can see.
python3 -c "
import json, sys

def resolved(path):
    '''What the MODEL sees: the file merged onto whatever default it inherits.

    Comparing the config FILE was the blind spot. configs_qweight inherits
    config/sca/default_model_cfg.json, which sets score_mode=centroid, so a baseline that
    simply omitted the key read as score_mode=None here and as CENTROID at run time -- every
    baseline scored with SCA's own aggregator, and this check said the config matched.'''
    c = json.load(open(path))['model_cfg']
    d = c.get('default')
    base = json.load(open(d)) if d else {}   # the default file is FLAT, not wrapped
    out = dict(base); out.update(c)
    return out

cfg = resolved('$CFG')
try:
    hps = json.load(open('workdir_pretrain/$ARM/log/hps.json'))['model_cfg']
except Exception as e:
    sys.exit('FATAL: cannot read workdir_pretrain/$ARM/log/hps.json (%s)' % e)
for key in ('model_type', 'score_mode', 'use_lora', 'sca_query_weighting', 'sca_frame_slots',
            'sca_tau_w'):
    want, got = cfg.get(key), hps.get(key)
    if (want or None) != (got or None):
        sys.exit('FATAL: $CFG resolves %s=%r but workdir_pretrain/$ARM trained with %r'
                 % (key, want, got))
print('model_type=%s score_mode=%s -- RESOLVED config matches the checkpoint'
      % (cfg.get('model_type'), cfg.get('score_mode')))
" || exit 2

# The ARM is part of the identity of a cell, not just the method name. Two different
# checkpoints of the same method -- gram_hyp2 at lr 2e-5 and h1_hypergram_paper at 1e-4 --
# would otherwise write to one directory, and since the config file is byte-identical between
# them the .done fingerprint matches too. The second run is then SKIPPED and the first
# checkpoint's numbers stand under the method's name forever. cell_is_done only started
# working recently, so this would have been a silent no-op before and is a live hazard now.
OUT="workdir/e1_repro/${METHOD}_${ARM}_${BENCH}"
cell_is_done "$OUT" "$CFG" && { echo "== [$METHOD/$BENCH] already done, skip"; exit 0; }
mkdir -p "$OUT"
claim_outdir "$OUT" || exit 2

echo "method : $METHOD   (arm $ARM)"
echo "ckpt   : $CKPT"
echo "bench  : $BENCH"
echo "outdir : $OUT"
echo "START=$(date +%T)"

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
EVAL_CKPT="$CKPT" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
  --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
  --master_port $((9500 + IDX)) \
  ./benchmark_eval/run_eval.py --config "$CFG" --output_dir "$OUT" 2>&1 \
  | { grep -v --line-buffered -E "$NOISE" || true; }
rc=${PIPESTATUS[0]}

if [ $rc -eq 0 ]; then cell_mark_done "$OUT" "$CFG"; echo "== [$METHOD/$BENCH] OK"; fi
echo "EXIT=$rc DONE $(date +%T)"
echo
echo "  python3 scripts/raw_vs_itm.py --root workdir/e1_repro --pivot"
exit $rc
