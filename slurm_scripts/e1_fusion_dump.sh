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
#SBATCH --job-name=e1_fusion
#SBATCH -o ./slurm_scripts/logs/e1_fusion_%j.out
#SBATCH -e ./slurm_scripts/logs/e1_fusion_%j.out
# Why the centroid's advantage does not reach the reported table, measured rather than argued.
#
# refine_score_matrix() allocates a ZERO matrix and writes the ITM probability into the top-k
# cells only. The dual-encoder score -- the centroid for SCA, the Gramian volume for GRAM --
# is never part of the final ranking; it only chooses which k clips the cross-encoder scores.
# So a better aggregator can move the reported number through exactly one channel: putting
# the ground truth inside the candidate set more often. Once recall@k saturates, that channel
# is shut and the aggregator is worth zero, however much better it is.
#
# That is the shape of what we measured: on DiDeMo the centroid leads the released GRAM
# checkpoint by +4.6 R@1 as an aggregator and the reported number moves -0.3. On AudioCaps,
# where retrieval is hardest and the candidate set cannot be saturated, +3.2 becomes +3.0 --
# nearly all of it transfers. Same method, opposite outcomes, explained by recall@k alone.
#
# This job re-runs the evals with SCA_DUMP_RERANK set, saving the dual and ITM score matrices
# per cell. Nothing about the reported metric changes -- the dump only records the two
# matrices that produced it. Everything after that is post hoc and free:
#
#   python3 scripts/sweep_score_fusion.py workdir/e1_fusion/*/dumps/rerank_*.pt
#
# which prints recall@k (the ceiling on any aggregator gain) and then sweeps score fusion,
# itm + w * z(dual), the variant BLIP uses and ALBEF/GRAM do not.
#
# Both arms are dumped because a fusion weight can only be adopted if it is applied to the
# baseline too. A w tuned on SCA alone and withheld from GRAM is not a result.
#
#   GRAM_RELEASED_CKPT=/leonardo_work/AIFAC_S07_041/HyperAlign/pretrained_weights/GRAM_pretrained_TVAS/ckpt/model_step_459.pt \
#     sbatch slurm_scripts/e1_fusion_dump.sh
#
#   SCA_ARM=workdir_pretrain/t1_lr1e4 GRAM_RELEASED_CKPT=... sbatch ...   # the lr 1e-4 arm
#
# Separate output root from e1_zs so the existing .done markers do not skip these cells.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
source "$(dirname "$0")/../scripts/cell_done.sh"
mkdir -p slurm_scripts/logs

best_ckpt() {
  local b; b=$(ls -t "$1"/ckpt/best_*.pt 2>/dev/null | head -1)
  [ -n "$b" ] || b=$(ls "$1"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1)
  echo "$b"
}

# The released GRAM checkpoint lives outside this repo, so there is nothing here to read it
# from and no default can be right; falling back to workdir_pretrain/gram would relabel our
# reproduction as the released model. The SCA side is the opposite case -- e4_transfer.sh
# records workdir_pretrain/sca as the arm behind the sca_* cells, so read it from there
# rather than making the caller retype a path.
[ -n "${GRAM_RELEASED_CKPT:-}" ] || { echo "FATAL: set GRAM_RELEASED_CKPT (the file behind the GRAM* rows)" >&2; exit 2; }
[ -f "$GRAM_RELEASED_CKPT" ] || { echo "FATAL: GRAM_RELEASED_CKPT=$GRAM_RELEASED_CKPT does not exist" >&2; exit 1; }
SCA_ARM="${SCA_ARM:-workdir_pretrain/sca}"
[ -d "$SCA_ARM" ] || { echo "FATAL: SCA_ARM=$SCA_ARM not found (ls workdir_pretrain to pick one)" >&2; exit 2; }
SCA_CKPT="${SCA_CKPT:-$(best_ckpt "$SCA_ARM")}"
[ -n "$SCA_CKPT" ] && [ -f "$SCA_CKPT" ] || { echo "FATAL: no checkpoint under $SCA_ARM/ckpt" >&2; exit 1; }
echo "GRAM released : $GRAM_RELEASED_CKPT"
echo "SCA ($SCA_ARM): $SCA_CKPT"

# msrvtt uses the configs_depth T-VAS config; configs_e1 has no sca_msrvtt.
cfg_for() {
  case "$1|$2" in
    sca\|msrvtt)  echo "benchmark_eval/configs_depth/sca_msrvtt_tvas.json" ;;
    sca\|*)       echo "benchmark_eval/configs_e1/sca_$2.json" ;;
    released\|*)  echo "benchmark_eval/configs_e1/gram_$2.json" ;;
  esac
}

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0
for arm in released sca; do
  [ "$arm" = released ] && ckpt="$GRAM_RELEASED_CKPT" || ckpt="$SCA_CKPT"
  for bench in msrvtt didemo activitynet vatex audiocaps; do
    cfg=$(cfg_for "$arm" "$bench")
    [ -f "$cfg" ] || { echo "== [$arm/$bench] SKIP: no config at $cfg" >&2; rc_all=2; continue; }
    out="workdir/e1_fusion/${arm}_${bench}"
    cell_is_done "$out" "$cfg" && { echo "== [$arm/$bench] already done, skip"; continue; }
    mkdir -p "$out/dumps"
    echo "== [$arm/$bench] START $(date +%T)"
    EVAL_CKPT="$ckpt" SCA_DUMP_RERANK="$out/dumps" \
      srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
      --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
      --master_port $((9950 + ${SLURM_JOB_ID:-$$} % 40)) \
      ./benchmark_eval/run_eval.py --config "$cfg" --output_dir "$out" 2>&1 \
      | { grep -v --line-buffered -E "$NOISE" || true; }
    rc=$?
    if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$arm/$bench] OK $(date +%T)"
    else echo "== [$arm/$bench] FAILED rc=$rc" >&2; rc_all=$rc; fi
  done
done

echo
echo "SELF-CHECK: these cells must reproduce the e1_zs numbers -- same checkpoints, same"
echo "            configs, only SCA_DUMP_RERANK added. Any drift means the dump changed the"
echo "            eval, and the fusion analysis would be built on a different model."
echo
echo "Next (no GPU needed):"
echo "  python3 scripts/sweep_score_fusion.py workdir/e1_fusion/*/dumps/rerank_*.pt"
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
