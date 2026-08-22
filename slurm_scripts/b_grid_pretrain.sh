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
#SBATCH --array=0-15
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
#   sbatch --array=6 slurm_scripts/b_grid_pretrain.sh      # T6: the frame set in the objective
#   sbatch --array=7-8 slurm_scripts/b_grid_pretrain.sh    # E1/E2: 1 and 2 epochs
#   sbatch --array=9-13 slurm_scripts/b_grid_pretrain.sh   # T7-T11: frame-set variants
#   sbatch --array=14-15 slurm_scripts/b_grid_pretrain.sh  # T12/T13: TRAINING frame parity
#
# T12/T13 fix a recipe gap that has been in every arm: we train on 2 frames and evaluate on
# 8, while GRAM trains on 8 (RECIPE_AUDIT.md:207 -- the audit only ever corrected the eval
# side). Worse, max_vision_sample_num is computed from the TRAIN block (utils/args.py:179),
# so pretraining learns a 2-position temporal embedding while the eval build wants 8 and
# nearest-neighbour interpolates the two up (general_module.py:130). Every number we hold
# came from a model whose temporal embedding was stretched 2 -> 8 at load. Both effects hit
# long videos hardest, which is exactly where our gaps are (ActivityNet, DiDeMo).
#
# T12 = 4 frames at batch 128, T13 = 8 frames at batch 64. Both are 512 frame-images, the
# vision load sca already trains at, and neither uses frame slots so there is no extra
# overhead. T13 reaches GRAM's 8-frame parity at half the in-batch negatives (256 vs 512);
# T12 keeps the full negative pool at half the frames. That trade is the point of running
# both.
#
# T9 is the ablation that matters most: T6 changes TWO things at once -- the frame axis and
# query weighting -- so a gain could come from either. T9 runs query weighting over
# modalities only, no frames. If T9 matches T6 the frames are irrelevant and the mechanism
# is the weighting; if T9 sits at b1 the frames carry it.
#
# T7-T11 set valid_freq 3 rather than 10. Ten in-training validations cost roughly 40
# minutes of wall clock each arm, and only the FINAL checkpoint is reported under the
# e1_final protocol -- three still locates the curve while letting more arms run per night.
#
# T7/T8 follow T6, which is the only arm with a positive signal: at matched steps it reads
# 53.0 and 53.8 where b1 reads 51.3 and 52.8, and it reaches b1's whole-run best (53.9) at
# step 1063 of 5295. T7 asks whether a RICHER set helps -- 4 frames at batch 64, the same
# 256 frame-images T6 proved fits, so more frames rather than more memory. T8 asks whether
# tau_w belongs at 0.05: 0.1 came from a post-hoc sweep over features trained under mean
# pooling, and the right temperature under training need not be the same.
#
# Both sit at T6's exact vision footprint. The 4-frame-at-batch-128 version died with CUDA
# OOM at 63.4 of 63.4 GiB, so frame count is traded against batch here, never added on top.
#
# E1/E2 test a regime nobody has: EVERY arm here has run 5 epochs, while GRAM's published
# recipe is ONE. Both start from the same VAST foundation checkpoint
# (model_step_204994.pt), and GRAM's released weights are model_step_459.pt on top of it --
# so we spend ~5.8x its adaptation steps for +0.9 R@1. Two overtraining signatures are
# already in the results: x1_xenc_full_lr2e5 fell from ~54.8 to 45.7 between its selected
# and final checkpoints, and sca's aggregator is far worse at the end of training than at
# its best step (DiDeMo 27.5 vs 32.8). Built on B1, the strongest arm measured.
#
# T6 trains what scripts/try_temporal_centroid.py measured on a finished checkpoint: the
# video enters the centroid as one slot per FRAME, weighted by the query. Measured post hoc
# that is worth +1.9 to +3.1 R@1 on the video pathway on all four benchmarks at tau_w=0.1,
# beating BOTH mean-pooling and max-over-frames -- and those features were trained under
# mean pooling, so it is a lower bound on what training with the set gives. Batch 128
# because b1 is the strongest arm measured, and 2 training frames because 4 died with CUDA
# OOM at 63.4 of 63.4 GiB -- 128 clips x 4 frames is 512 frame-images through ViT-g, double
# the 256 b1 proved fits. Two frames still make a SET rather than a pooled vector, and eval
# keeps 8 since the centroid is arity-invariant and the counts need not match.
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

ARMS=(B1_bs128_r8 B2_bs128_r32 B3_bs512_r8 B4_bs512_r32 B5_bs128_xenc B6_bs512_xenc T6_frameset E1_bs128_ep1 E2_bs128_ep2 T7_frameset_4f T8_frameset_tau005 T9_qweight_only T10_frameset_bs256 T11_frameset_tau02 T12_qw_4frames T13_qw_8frames)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[ -n "$IDX" ] || { echo "FATAL: no array index. sbatch this, or pass 0-3 to run one arm." >&2; exit 2; }
ARM="${ARMS[$IDX]:-}"
[ -n "$ARM" ] || { echo "FATAL: index $IDX out of range (0-15)" >&2; exit 2; }

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
