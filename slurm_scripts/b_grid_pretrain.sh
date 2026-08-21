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
#SBATCH --job-name=b_grid
#SBATCH --array=0-5
#SBATCH -o ./slurm_scripts/logs/b_grid_%A_%a.out
#SBATCH -e ./slurm_scripts/logs/b_grid_%A_%a.out
# Batch size x LoRA capacity, from the best configuration we have.
#
# The reference arm is LoRA at lr 2e-5 (config/sca/pretrain_cfg/sca_pretrain.json), which
# wins 4 of 5 benchmarks against SCA at 1e-4 -- the latter leads only on MSR-VTT, the
# benchmark it was selected on. Rank was already swept at batch 256 (A6_lora_r2..r64), and
# batch size has never been varied: every one of the 44 trained arms used 256. These four
# runs cross the two axes, and together with sca_pretrain (256/r8) and A6_lora_r32 (256/r32)
# they complete a 3x2 grid:
#
#            r=8            r=32
#   bs 128   B1 (new)       B2 (new)
#   bs 256   sca (trained)  A6_lora_r32 (trained)
#   bs 512   B3 (new)       B4 (new)
#
# lr is held at 2e-5 across the grid. That does mean the batch comparison carries an
# effective-learning-rate difference with it, which is the honest caveat -- but the goal is
# the best single (lr, batch, rank) point to report everywhere, not an isolated batch-size
# effect, and a per-batch lr would reintroduce exactly the multi-lr selection we are trying
# to avoid.
#
# Epochs stay at 5 for every arm, so all six cells see identical data the same number of
# times; only the optimizer-step count moves (5290 / 2645 / 1322). That is what changing
# batch size means, and audit_training_gaps.py will correctly report the step targets as
# differing -- expected here, and the reason that check names the field it disagrees on.
#
# A job array: four independent jobs, one arm each, so a failure or an OOM in one does not
# take the others down. The 512 arms run 128 clips/GPU at 2 training frames; if they OOM it
# is the batch size, not a bug, and B1/B2 are unaffected.
#
# B5/B6 combine the two axes that independently worked. Measured against the released GRAM
# checkpoint: b1 (batch 128) wins MSR-VTT (+1.2) and VATEX (+0.5), b3 (batch 512) wins
# AudioCaps (+4.4), and x1_xenc_full_lr2e5 (cross-encoder trainable, batch 256) wins DiDeMo
# (+0.8) and is closest on ActivityNet. Batch size and cross-encoder training improve
# DIFFERENT benchmarks by different mechanisms and have never been combined, so B5 = batch
# 128 + trainable cross-encoder and B6 = batch 512 + trainable cross-encoder.
#
#   sbatch slurm_scripts/b_grid_pretrain.sh                # all six
#   sbatch --array=4-5 slurm_scripts/b_grid_pretrain.sh    # just the new combined pair
#
# Resubmit to resume: each arm restarts from its own optimizer checkpoint.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
: "${DATA_ROOT:?export DATA_ROOT}"
: "${WORK_ROOT:?export WORK_ROOT}"
: "${SCA_CACHE_ROOT:?export SCA_CACHE_ROOT (must hold s_star_150k.pt)}"
MODELS_DIR="${MODELS_DIR:-$WORK_ROOT/sca_models}"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" \
  || { echo "FATAL: $MODELS_DIR/env.sh not found -- run scripts/prefetch_models.py first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
mkdir -p slurm_scripts/logs

ARMS=(B1_bs128_r8 B2_bs128_r32 B3_bs512_r8 B4_bs512_r32 B5_bs128_xenc B6_bs512_xenc)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[ -n "$IDX" ] || { echo "FATAL: no array index. sbatch this, or pass 0-3 to run one arm." >&2; exit 2; }
ARM="${ARMS[$IDX]:-}"
[ -n "$ARM" ] || { echo "FATAL: index $IDX out of range (0-5)" >&2; exit 2; }

CFG="config/sca/ablations/${ARM}.json"
[ -f "$CFG" ] || { echo "FATAL: $CFG not found" >&2; exit 2; }
OUT="workdir_pretrain/$(echo "$ARM" | tr 'A-Z' 'a-z')"

# the S* cache is built once on a login node; without it the semantic loss has no targets
S_STAR="$SCA_CACHE_ROOT/s_star_150k.pt"
[ -f "$S_STAR" ] || { echo "FATAL: $S_STAR missing -- build it before submitting:" >&2
  echo "  python3 data/semantic_targets.py --annotation_json \$DATA_ROOT/vast27m_150k/annotations150k.json --out_path $S_STAR" >&2
  exit 2; }

RESUME=""
if ls "$OUT"/ckpt/optimizer_step_*.pt >/dev/null 2>&1; then
  RESUME="--resume true"; echo "RESUME: $OUT has an optimizer checkpoint -> continue"
else
  echo "FRESH: $OUT"
fi

echo "arm    : $ARM"
echo "config : $CFG"
echo "outdir : $OUT"
python3 -c "
import json;c=json.load(open('$CFG'))
print('batch  : %d  (epochs %d)' % (c['data_cfg']['train'][0]['batch_size'], c['data_cfg']['train'][0]['epoch']))
print('rank   : %d  (alpha %d)' % (c['model_cfg']['lora_r_vision'], c['model_cfg']['lora_alpha']))
print('lr     : %s' % c['run_cfg']['learning_rate'])"
echo "START=$(date +%T)"

srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 \
  --master_port $((9700 + IDX)) \
  ./run.py --config "$CFG" --output_dir "$OUT" --checkpointing true $RESUME 2>&1
rc=$?
echo "EXIT=$rc DONE $(date +%T)"
if [ $rc -ne 0 ]; then
  echo "[$ARM] failed. If the log ends in CUDA out of memory this is the batch size --" >&2
  echo "       the 512 arms run 128 clips/GPU against 64 for the reference arm." >&2
fi
exit $rc
