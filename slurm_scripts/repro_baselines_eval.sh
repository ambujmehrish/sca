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
# Every competing aggregator, trained and evaluated by us, on all five benchmarks.
#
# The published table mixes numbers from four evaluation environments, and we have measured
# that environments differ: the SAME released GRAM checkpoint reads 54.8 in the paper and 52.5
# here on MSR-VTT, and 83.5 there against 90.0 here on VATEX. Cross-environment differences of
# one or two points are therefore not interpretable, which is most of the range this field
# competes in.
#
# This table removes that. Same VAST foundation checkpoint, same 150k training set, same
# schedule, same eval data blocks, same 8 frames, same rerank depth 50 -- only the aggregation
# geometry differs:
#
# MAIN TABLE rows -- methods as their authors proposed them:
#   pmrl        leading eigenvalue        (lambda_1 of the Gram matrix)
#   hypergram   hyperbolic Gram           (our reimplementation, see below)
#   sca         query-weighted centroid   (ours)
# plus GRAM's released checkpoint, already measured in workdir/e1_zs (released_* cells).
#
# APPENDIX row -- not a published method:
#   gram_lora   Gramian volume + LoRA
#
# gram_lora is our construction, not something GRAM proposes, so it does not belong in a
# comparison table: a reader would reasonably ask who claimed it. Its job is the control that
# separates the two variables our recipe changes at once -- SCA is centroid AND adapter, so
# volume-plus-adapter is what says whether the gain is the geometry or the LoRA. That is an
# ablation argument and it belongs in the appendix beside the other ablations.
#
# A difference among the main rows is a difference in the aggregator, which is the
# algorithmic claim.
#
# HYPERGRAM: read experiments/results/HYPERGRAM_STATUS.md before quoting any number here.
#
# The default arm for this row is gram_hyp2, which was trained at lr 2e-5 and fell to 37.4.
# That figure is NOT evidence about HyperGRAM and must not be cited as a reproduction result.
# 2e-5 came from the HyperAlign trunk, and wave4/ANALYSIS.md had already found exactly that to
# be the recipe defect for OUR method -- SCA went from 53.5 to 54.9 when moved to 1e-4. The
# same correction was never applied to the HyperGRAM arm before its number was recorded.
#
# H1_hypergram_paper (b_grid --array=41) is the same v2 reading at lr 1e-4, batch 128: the
# recipe their paper uses. Until it has run there is no HyperGRAM reproduction at all -- not a
# failed one, an unrun one. Point this row at it once it exists:
#
#   SCA_REPRO_HYP_ARM=h1_hypergram_paper sbatch --array=5-9 slurm_scripts/repro_baselines_eval.sh
#
# Whatever it reads, the row is OUR REIMPLEMENTATION and never HyperGRAM's performance: their
# code is not released and the hyperbolic branch admits two readings. Their published numbers
# stand as cited in the main table.
#
#   sbatch --array=0-9 slurm_scripts/repro_baselines_eval.sh      # the MAIN-TABLE rows
#   sbatch --array=10-14 slurm_scripts/repro_baselines_eval.sh    # gram_lora, appendix only
#   sbatch slurm_scripts/repro_baselines_eval.sh                  # all 15 cells
#   sbatch --array=0-4 slurm_scripts/repro_baselines_eval.sh      # pmrl only
#   SCA_REPRO_HYP_ARM=gram_hyp sbatch --array=5-9 ...             # the v1 reading instead
set -uo pipefail
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
cfg = json.load(open('$CFG'))['model_cfg']
try:
    hps = json.load(open('workdir_pretrain/$ARM/log/hps.json'))['model_cfg']
except Exception as e:
    sys.exit('FATAL: cannot read workdir_pretrain/$ARM/log/hps.json (%s)' % e)
want, got = cfg.get('model_type'), hps.get('model_type')
if want != got:
    sys.exit('FATAL: $CFG scores model_type=%r but workdir_pretrain/$ARM was trained as %r'
             % (want, got))
if cfg.get('score_mode') != hps.get('score_mode'):
    sys.exit('FATAL: $CFG uses score_mode=%r, the arm trained with %r'
             % (cfg.get('score_mode'), hps.get('score_mode')))
print('model_type=%s score_mode=%s -- config matches the checkpoint' % (want, cfg.get('score_mode')))
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
