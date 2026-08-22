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
#SBATCH --job-name=itmfrz
#SBATCH --array=0-4
#SBATCH -o ./slurm_scripts/logs/itmfrz_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/itmfrz_%A_%a.out
# Does the ITM reranker lose SCA's lead because our adapters drifted it?
#
# The reported number comes from two stages with different training histories. Stage 1, the
# contrastive scorer, is what LoRA was trained for. Stage 2 is a cross-encoder plus an ITM
# head that came pretrained in the VAST foundation checkpoint and was never trained again --
# but the retrieval loss reaches its BERT through the SAME multimodal_encoder adapters, and
# its condition_feats come from the adapted vision and audio encoders. Every LoRA step drifts
# stage 2 away from the calibration its own frozen head was fitted to, for a gradient that is
# not the ITM objective.
#
# The measurement says exactly that, on all five benchmarks. T9 against the released GRAM
# checkpoint, aggregator score then reported ITM score:
#
#   MSR-VTT      45.2 vs 38.7  (+6.5)  ->  54.8 vs 52.5  (+2.3)
#   DiDeMo       34.2 vs 28.2  (+6.0)  ->  51.5 vs 50.7  (+0.8)
#   ActivityNet  34.4 vs 31.0  (+3.4)  ->  55.8 vs 56.3  (-0.5)
#   VATEX        81.7 vs 75.6  (+6.1)  ->  90.5 vs 90.0  (+0.5)
#   AudioCaps    27.1 vs 22.9  (+4.2)  ->  35.2 vs 32.2  (+3.0)
#
# Better candidates on every benchmark, and 1.2 to 5.6 points of that lead gone after
# reranking. One stage, the same behaviour five times.
#
# itm_lora_off runs stage 2 through the frozen backbone: the dual encoder keeps its adapters
# and still supplies the candidates, the reranker scores them with the weights its head was
# fitted to. Inference only -- nothing is retrained, and the checkpoints are the T9 finals
# already on disk.
#
# WHAT THIS IS NOT. These checkpoints were TRAINED with the adapters in the ITM branch, so
# evaluating without them is a train/test mismatch by construction. This is a diagnostic that
# answers one question -- is adapter drift what the reranker costs us? -- and it is not the
# recipe. If it wins, the follow-up is to retrain with itm_lora_off set (the flag applies to
# the training ITM loss too, so it is one key in the pretrain config), and THAT arm is what
# a paper reports.
#
# Cost: the flag adds a second vision/audio encoder pass for the frozen condition_feats, so a
# cell takes roughly twice as long as the same cell in fs_eval. One benchmark per array task,
# so a slow benchmark does not hold up the rest.
#
#   sbatch slurm_scripts/itm_frozen_eval.sh                 # all five
#   sbatch --array=2 slurm_scripts/itm_frozen_eval.sh       # activitynet alone (the -0.5)
#   SCA_ITM_ARM=t8_frameset_tau005 sbatch slurm_scripts/itm_frozen_eval.sh
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" \
  || { echo "FATAL: $MODELS_DIR/env.sh not found -- run scripts/prefetch_models.py first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
source "$(dirname "$0")/../scripts/cell_done.sh"
mkdir -p slurm_scripts/logs

BENCHES=(msrvtt didemo activitynet vatex audiocaps)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[ -n "$IDX" ] || { echo "FATAL: no array index. sbatch this, or pass 0-4 to run one benchmark." >&2; exit 2; }
BENCH="${BENCHES[$IDX]:-}"
[ -n "$BENCH" ] || { echo "FATAL: index $IDX out of range (0-4)" >&2; exit 2; }

ARM="${SCA_ITM_ARM:-t9_qweight_only}"
# the FINAL checkpoint, matching e1_final_ckpt.sh and fs_eval -- never best_*.pt, which
# save_best selects on the aggregator score rather than on the reported metric
CKPT=$(ls "workdir_pretrain/$ARM"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1)
[ -n "$CKPT" ] && [ -f "$CKPT" ] || {
  echo "FATAL: no model_step_*.pt under workdir_pretrain/$ARM -- is the arm trained?" >&2; exit 2; }

CFG="benchmark_eval/configs_qweight_itmfrozen/sca_${BENCH}.json"
[ -f "$CFG" ] || { echo "FATAL: $CFG not found" >&2; exit 2; }
# refuse to run the flag off under a name that claims it is on: a config that lost the key
# would produce a plain fs_eval number and land in a directory labelled itm_frozen
python3 -c "
import json,sys
c=json.load(open('$CFG'))
if not c['model_cfg'].get('itm_lora_off'):
    sys.exit('FATAL: $CFG does not set itm_lora_off -- this cell would be a duplicate of fs_eval')
if not c['model_cfg'].get('use_lora'):
    sys.exit('FATAL: $CFG has use_lora=false -- there would be no adapters to switch off')
" || exit 2

OUT="workdir/e1_itmfrozen/${ARM}_${BENCH}"
cell_is_done "$OUT" "$CFG" && { echo "== [$ARM/$BENCH] already done, skip"; exit 0; }
mkdir -p "$OUT"

echo "arm    : $ARM"
echo "ckpt   : $CKPT"
echo "bench  : $BENCH"
echo "config : $CFG   (itm_lora_off=true)"
echo "outdir : $OUT"
echo "START=$(date +%T)"

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
EVAL_CKPT="$CKPT" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
  --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
  --master_port $((9300 + IDX)) \
  ./benchmark_eval/run_eval.py --config "$CFG" --output_dir "$OUT" 2>&1 \
  | { grep -v --line-buffered -E "$NOISE" || true; }
rc=${PIPESTATUS[0]}

if [ $rc -eq 0 ]; then cell_mark_done "$OUT" "$CFG"; echo "== [$ARM/$BENCH] OK"; fi
echo "EXIT=$rc DONE $(date +%T)"
echo
echo "compare against the same arm WITHOUT the flag:"
echo "  python3 scripts/raw_vs_itm.py --root workdir/e1_itmfrozen --pivot"
echo "  python3 scripts/raw_vs_itm.py --root workdir/e1_frames   --pivot"
exit $rc
