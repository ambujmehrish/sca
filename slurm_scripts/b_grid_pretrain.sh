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
#   sbatch --array=16 slurm_scripts/b_grid_pretrain.sh     # T14: adapters off the reranker
#   sbatch --array=17-18 slurm_scripts/b_grid_pretrain.sh  # S1/S2: seeds on the reported recipe
#   sbatch --array=19-29 slurm_scripts/b_grid_pretrain.sh  # G1-G10: capacity, lr, objective
#   sbatch --array=30-32 slurm_scripts/b_grid_pretrain.sh  # X3-X5: cross-encoder, done properly
#   sbatch --array=33-40 slurm_scripts/b_grid_pretrain.sh  # X6-X13: HOW MUCH to move it
#   sbatch --array=46 slurm_scripts/b_grid_pretrain.sh     # G11: T9 with NO masked training
#
# G11_train_nomask holds mask_p_full at 1.0 for the whole run: no masked view is ever drawn,
# so the main alignment loss trains on the full-arity centroid only. This is the arm G10 is
# not: G10 zeroes beta (the L_mask agreement term) but the mask sampler still feeds present_M
# into L_align, so G10 trains WITH masked views and G11 trains WITHOUT them. At p_full=1 the
# two centroids coincide and l_mask is exactly zero in value and gradient, so the two
# mask_p_full keys are the arm's only live difference from T9.
#   (no HyperGRAM arm here: their code is released, so the reproduction runs THEIR repo
#    with THEIR config -- see experiments/results/HYPERGRAM_STATUS.md. H1 was removed because
#    it claimed their recipe while carrying GRAM's: lr 1e-4 not 5e-5, 5 epochs not 1, and
#    without the subtitle task theirs trains.)
#
# X6-X13 attack the term the reported metric actually turns on. R@1 factors as candidate
# recall times the reranker's accuracy on those candidates, and the second is the small one:
# 61.3% on MSR-VTT against 89.4% recall. Reaching HyperGRAM's published 56.6 from our 54.8
# needs that 61.3 to become 63.3; ActivityNet needs 59.2 -> 61.8. Recall is not the constraint
# and never was -- we already beat the released GRAM checkpoint on recall on all five.
#
# X3-X5 showed fine-tuning the cross-encoder at the base rate for the full schedule DESTROYS
# it: 51.4 / 51.1 / 50.9 against 54.8 frozen. But all three are HIGHEST at their first
# validation (step 1776) and falling, and GRAM's released weights are model_step_459. We have
# never looked at the range GRAM actually uses. "Freeze it" was the wrong lesson; "how much"
# is the question.
#
# Two mechanisms drive the forgetting. Scale: 5330 steps on 150k clips against the 27M this
# component was pretrained on. Modality mix: the ITM loss trains on condition_feats_va
# (gram.py:732, hardcoded) and our training set has no subtitles at all, while MSR-VTT and
# VATEX are scored with tvas -- so fine-tuning erases a subtitle pathway that no gradient in
# this recipe can restore, on exactly the benchmark where the gap sits.
#
# X6/X7  one epoch, so the whole run lives near GRAM's step count.
# X8-X10 the full schedule, with the cross-encoder on its own rate 10-40x below the heads.
# X11/X12 only the top 2 or 4 BERT layers move; the lower layers keep VAST's representation.
# X13    itm_ratio 0.5 rather than 0.1 -- if this component is what we are training, ask
#        whether it is trained hard ENOUGH relative to how fast it forgets.
#
# Every one validates ten times instead of three. The three-point schedule cannot see below
# step 1776, and the entire hypothesis is that the peak is below it.
#
# X3-X5 REDO the trainable-cross-encoder experiment, which was never validly run.
#
# B5, B6, X1 and X2 all set lora_freeze_multimodal=false and left lora_r_text=8. That trains
# the cross-encoder's W_q and W_v TWICE per step -- the base weight at learning_rate and a
# rank-8 adapter (alpha/r = 2) at 0.1 x learning_rate, on the same matrices -- while the key
# projection, output.dense and the FFN in the same layer get only the base update. The layer's
# attention then moves at a different effective rate from the rest of it. That is not full
# fine-tuning and it is not what GRAM does.
#
# The symptom was there to read: x1_xenc_full_lr2e5 validated 52.6 / 45.8 / 50.3 / 48.7 / 40.7
# / 40.3 / 43.7 / 49.5 / 49.8 / 49.2. Oscillation, not decay. It was called overtraining and
# used to conclude that training the cross-encoder does not work -- a conclusion drawn from a
# defect. build_optimizer.py now refuses the combination outright.
#
# X3-X5 set lora_r_text = 0 with lora_freeze_multimodal = false: vision and audio keep their
# rank-8 adapters, and the cross-encoder is fine-tuned cleanly with one parameterization, as
# GRAM and HyperGRAM do. multimodal_encoder is both the text tower and the cross-encoder, so
# this unfreezes both -- which is also what GRAM full-FT does.
#
# X3 uses 2e-5, GRAM's own pretraining learning rate, so it is the matched control. X4 and X5
# halve and quarter it: GRAM trains at batch 256 and this recipe at 128, and a pretrained BERT
# needs less to move than a fresh head does.
#
# G1-G10 are each T9 with ONE key changed. The recipe has converged -- t9 reads 54.3 / 54.8 /
# 54.8 over its three validations, flat across the last third -- so a longer schedule is not
# the gap and none of these lengthens it. What has never been crossed is capacity WITH query
# weighting: b2 (r32, uniform centroid) peaked 54.4 against b1's 53.9 at the same batch, and
# T9 is b1 plus query weighting at 54.8. G1-G3 sweep rank on top of the weighting, holding
# alpha at 2r so the adapter scale is constant and the comparison is rank alone; G2b repeats
# r32 at alpha 16 to match b2 exactly, which is the only way to separate a rank effect from
# the scale change b2 silently carried.
#
# G4/G5 bracket the learning rate. 2e-5 beat 1e-4 on 4 of 5 benchmarks, so the optimum is at
# or below 2e-5 and neither side of it has been probed at this recipe.
#
# G6-G10 re-check the auxiliary losses. They were last tuned under the UNIFORM centroid at
# batch 256; query weighting changes what the alignment term does, so a weight that helped
# then need not help now. These double as the paper's ablation table, which has to be run at
# the reported configuration rather than at an older one.
#
# S1/S2 are T9 with run_cfg.seed changed and nothing else. They exist because the table has no
# error bars, and the eval-side floor measured from repeated evals (0.2 R@1, raw_vs_itm.py) is
# a LOWER bound: it shares a checkpoint, so it says nothing about seed-to-seed training
# variance, which is the larger term and the one a reviewer will ask about. Three runs of the
# reported configuration give a mean and a range for every row of the main table.
#
# These take priority over any new hypothesis. A margin without an error bar is not a result,
# and every SCA row currently has n=1.
#
# T14 = T9 plus itm_lora_off, differing in that one key and nothing else. It is the RECIPE
# version of what slurm_scripts/itm_frozen_eval.sh measures as a diagnostic: the reranker is a
# pretrained cross-encoder and a frozen ITM head that was never trained here, yet the retrieval
# loss reaches its BERT through the same multimodal_encoder adapters, so every step drifts it
# away from the calibration its head was fitted to. T9 leads the released GRAM checkpoint on
# the aggregator on all five benchmarks (+3.4 to +6.5) and keeps only +2.3/+0.8/-0.5/+0.5/+3.0
# after reranking. In T14 the adapters take no gradient from the ITM branch and the branch is
# scored on the weights it was fitted on, so train and test agree.
#
# ORDER MATTERS: run itm_frozen_eval.sh on the existing T9 checkpoints FIRST. That costs eval
# only and tells you whether adapter drift is the cost at all. Do not spend a 24h training
# slot on T14 before that reads out.
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
# We have already cd'd to CODE_DIR, so resolve the helper from there. $0 under sbatch is a
# COPY of this script in Slurm's spool directory, not the file in the repo, so
# "$(dirname "$0")/.." can point somewhere with no scripts/ at all. The eval launchers have
# always used that form and happen to work, but a source that silently fails leaves
# claim_outdir undefined -- and `claim_outdir "$OUT" || exit 2` then kills the job in seconds
# with "command not found", which reads as the experiment failing rather than the launcher.
HELPER="${CODE_DIR:-.}/scripts/cell_done.sh"
[ -f "$HELPER" ] || HELPER="$(dirname "$0")/../scripts/cell_done.sh"
# shellcheck source=/dev/null
source "$HELPER" || { echo "FATAL: cannot source $HELPER" >&2; exit 2; }
command -v claim_outdir >/dev/null || {
  echo "FATAL: sourced $HELPER but claim_outdir is not defined." >&2; exit 2; }

ARMS=(B1_bs128_r8 B2_bs128_r32 B3_bs512_r8 B4_bs512_r32 B5_bs128_xenc B6_bs512_xenc T6_frameset E1_bs128_ep1 E2_bs128_ep2 T7_frameset_4f T8_frameset_tau005 T9_qweight_only T10_frameset_bs256 T11_frameset_tau02 T12_qw_4frames T13_qw_8frames T14_itm_frozen S1_t9_seed51 S2_t9_seed52 G1_r16_qw G2_r32_qw G2b_r32_a16_qw G3_r64_qw G4_lr5e5 G5_lr1e5 G6_lambda0 G7_lambda03 G8_sem0 G9_concept0 G10_mask0 X3_xenc_clean_lr2e5 X4_xenc_clean_lr1e5 X5_xenc_clean_lr5e6 X6_xenc_1ep_lr2e5 X7_xenc_1ep_lr5e6 X8_xenclr_1e6 X9_xenclr_2e6 X10_xenclr_5e7 X11_xenc_top2 X12_xenc_top4 X13_xenclr_2e6_itm05 R1_itm_vas R2_itm_top50 R3_itm_top50_n4 R4_itm_vas_top50_n4 F1_t9_fullft G11_train_nomask)
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[ -n "$IDX" ] || { echo "FATAL: no array index. sbatch this, or pass 0-3 to run one arm." >&2; exit 2; }
ARM="${ARMS[$IDX]:-}"
[ -n "$ARM" ] || { echo "FATAL: index $IDX out of range (0-$((${#ARMS[@]} - 1)))" >&2;
  exit 2; }

# Arms live in ablations/ (the sweep) or reranker/ (the stage-2 arms). Searched rather than
# hardcoded, and a name present in BOTH is fatal: two configs answering to one arm name is
# how a cell gets scored with a geometry it was not trained with, which audit_eval_geometry
# has already caught 25 times.
CFG=""
for d in config/sca/ablations config/sca/reranker; do
  if [ -f "$d/${ARM}.json" ]; then
    [ -z "$CFG" ] || { echo "FATAL: $ARM exists in more than one config directory:" >&2
      echo "         $CFG" >&2; echo "         $d/${ARM}.json" >&2
      echo "       Rename one -- an arm name must identify exactly one config." >&2; exit 2; }
    CFG="$d/${ARM}.json"
  fi
done
[ -n "$CFG" ] || { echo "FATAL: no config for arm $ARM in config/sca/{ablations,reranker}" >&2
  exit 2; }
OUT="workdir_pretrain/$(echo "$ARM" | tr 'A-Z' 'a-z')"

# Refuse to start if another live job is already writing this directory. A duplicate
# submission is otherwise silent: both jobs log normal progress while overwriting each
# other's checkpoints. See scripts/cell_done.sh for the incident this comes from.
claim_outdir "$OUT" || exit 2

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
# Every line independent, and nothing indexed that an arm is allowed not to have. The
# previous version read c['model_cfg']['lora_r_vision'] directly, so a non-LoRA arm (H1, the
# HyperGRAM reproduction) raised KeyError on that line and the block died -- taking the
# LEARNING RATE line with it. That is the one field H1 exists to verify, and its absence read
# as a display quirk rather than as the check failing.
python3 -c "
import json
c = json.load(open('$CFG'))
m, r, d = c['model_cfg'], c['run_cfg'], c['data_cfg']['train'][0]
print('model  : %s' % m.get('model_type', 'sca'))
print('batch  : %d  (epochs %d)' % (d['batch_size'], d['epoch']))
print('lr     : %s' % r.get('learning_rate'))
# The stage-2 knobs, printed for every arm. An arm whose name promises hard negatives
# and whose config does not carry them is the failure mode this whole block exists for.
print('itm    : neg_topk=%s num_neg=%s condition=%s' % (
    m.get('itm_neg_topk', 0) or 'all', m.get('itm_num_neg', 1) or 1,
    m.get('itm_condition_key', 'va')))
if m.get('use_lora'):
    print('rank   : %s  (alpha %s)' % (m.get('lora_r_vision'), m.get('lora_alpha')))
    print('xenc   : freeze_mm=%s r_text=%s xenc_lr=%s top_layers=%s'
          % (m.get('lora_freeze_multimodal', True), m.get('lora_r_text'),
             r.get('xenc_lr'), m.get('xenc_train_layers', 0)))
else:
    print('rank   : (no adapter -- full fine-tune)')
for k in ('sca_query_weighting', 'sca_tau_w', 'sca_frame_slots', 'hyp_use_prenorm', 'seed'):
    v = m.get(k, r.get(k))
    if v is not None:
        print('%-7s: %s' % (k[:7], v))"
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
