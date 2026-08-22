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
#SBATCH --job-name=itmqw
#SBATCH --array=0-24
#SBATCH -o ./slurm_scripts/logs/itmqw_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/itmqw_%A_%a.out
# Query-weighted reranking, swept over gamma. Inference only -- nothing is retrained.
#
# THE NUMBER THIS TARGETS. The reported metric factors exactly:
#
#     R@1  =  candidate recall@50  x  P(reranker ranks the GT first | GT is a candidate)
#
# Both terms are measured (scripts/rerank_conversion.py). On AudioCaps they are 89.9 and
# 39.2. The second is the small one on every benchmark, and it is the one SCA does not touch:
# query weighting is applied to the contrastive features and to nothing else, so it decides
# WHICH 50 clips the cross-encoder sees and then has no say in the ranking that is reported.
#
# Restated against the target: to reach HyperGRAM's published 56.6 on MSR-VTT from our 54.8 at
# recall 89.4, the reranker's conditional accuracy has to go from 61.3% to 63.3%. On
# ActivityNet, 59.2% to 61.8%. Those are the gaps; recall is not where they live. Our
# AudioCaps lead over the released GRAM checkpoint already comes from this term and not from
# recall -- 41.2% against 37.7% on the 597 queries both methods reach, while the queries only
# SCA recovers net to roughly zero.
#
# WHAT THE FLAG DOES.
#
#     score = (1 - gamma) * ITM(joint)  +  gamma * sum_m w_m(t, clip) * ITM(modality m)
#
# w_m is the same query-conditioned weight the centroid uses, so both stages weigh the
# modalities by one rule instead of two unrelated ones. Each per-modality pass feeds the
# cross-encoder a single modality's condition_feats -- exactly the input the tv and ta
# pretraining tasks give it -- so no feature is rescaled and the frozen ITM head is never
# taken out of distribution. The weighting acts on the head's OUTPUTS, where it cannot break a
# calibration. That is the difference from scaling condition_feats, which would.
#
# gamma = 0 is in the sweep on purpose. It must reproduce the fs_eval number for the same arm
# EXACTLY. If it does not, the extra plumbing has perturbed the baseline and no other gamma in
# the sweep means anything -- check that cell before reading the rest.
#
# Cost: one extra cross-encoder pass per modality, so a tva cell is ~3x an fs_eval cell and a
# tvas cell ~4x. Reranking dominates eval, and gamma=0 skips the extra passes entirely.
#
#   sbatch slurm_scripts/itm_qw_eval.sh                    # 5 gammas x 5 benchmarks
#   sbatch --array=0-4 slurm_scripts/itm_qw_eval.sh        # gamma=0 first: the null check
#   SCA_ITMQW_ARM=g2_r32_qw sbatch slurm_scripts/itm_qw_eval.sh
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" \
  || { echo "FATAL: $MODELS_DIR/env.sh not found -- run scripts/prefetch_models.py first" >&2; exit 1; }
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

GAMMAS=(g000 g030 g050 g070 g100)
BENCHES=(msrvtt didemo activitynet vatex audiocaps)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[ -n "$IDX" ] || { echo "FATAL: no array index. sbatch this, or pass 0-24 for one cell." >&2; exit 2; }
GAMMA="${GAMMAS[$((IDX / 5))]:-}"
BENCH="${BENCHES[$((IDX % 5))]:-}"
[ -n "$GAMMA" ] && [ -n "$BENCH" ] || { echo "FATAL: index $IDX out of range (0-24)" >&2; exit 2; }

ARM="${SCA_ITMQW_ARM:-t9_qweight_only}"
CKPT=$(ls "workdir_pretrain/$ARM"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1)
[ -n "$CKPT" ] && [ -f "$CKPT" ] || {
  echo "FATAL: no model_step_*.pt under workdir_pretrain/$ARM" >&2; exit 2; }

CFG="benchmark_eval/configs_qweight_itmqw/${GAMMA}/sca_${BENCH}.json"
[ -f "$CFG" ] || { echo "FATAL: $CFG not found" >&2; exit 2; }
# A config that lost the key, or that is not using the query-weighted centroid, would run the
# ordinary reranker and write the result into a directory named for this experiment.
python3 -c "
import json,sys
c=json.load(open('$CFG'))['model_cfg']
if 'sca_itm_qw_gamma' not in c: sys.exit('FATAL: $CFG has no sca_itm_qw_gamma')
if not c.get('sca_query_weighting'): sys.exit('FATAL: $CFG is not using the query-weighted centroid')
want = int('${GAMMA#g}') / 100.0        # the directory name IS the gamma; they must agree
if abs(c['sca_itm_qw_gamma'] - want) > 1e-9:
    sys.exit('FATAL: $CFG sets gamma %.3f but sits in directory ${GAMMA} (%.3f). The results '
             'would be filed under the wrong gamma.' % (c['sca_itm_qw_gamma'], want))
print('gamma = %.2f' % c['sca_itm_qw_gamma'])
" || exit 2

OUT="workdir/e1_itmqw/${ARM}_${GAMMA}_${BENCH}"
cell_is_done "$OUT" "$CFG" && { echo "== [$GAMMA/$BENCH] already done, skip"; exit 0; }
mkdir -p "$OUT"
claim_outdir "$OUT" || exit 2

echo "arm    : $ARM"
echo "ckpt   : $CKPT"
echo "cell   : $GAMMA / $BENCH"
echo "outdir : $OUT"
echo "START=$(date +%T)"

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
EVAL_CKPT="$CKPT" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
  --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
  --master_port $((9400 + IDX)) \
  ./benchmark_eval/run_eval.py --config "$CFG" --output_dir "$OUT" 2>&1 \
  | { grep -v --line-buffered -E "$NOISE" || true; }
rc=${PIPESTATUS[0]}

if [ $rc -eq 0 ]; then cell_mark_done "$OUT" "$CFG"; echo "== [$GAMMA/$BENCH] OK"; fi
echo "EXIT=$rc DONE $(date +%T)"
echo
echo "read it with:  python3 scripts/itm_qw_sweep.py"
echo "the g000 row MUST match workdir/e1_frames for the same arm, or nothing else counts."
exit $rc
