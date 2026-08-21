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
#SBATCH --job-name=e1_final
#SBATCH -o ./slurm_scripts/logs/e1_final_%j.out
#SBATCH -e ./slurm_scripts/logs/e1_final_%j.out
# Re-evaluate on the FINAL checkpoint instead of the aggregator-selected one.
#
# workdir_pretrain/sca/ckpt holds two different models:
#   best_ret%tvas--msrvtt_ret_ret_area_forward.pt   <- what every eval so far has used
#   model_step_2649.pt                              <- the end of the schedule
#
# The name of the first one is the whole problem: save_best selects on
# get_best_name() == 'volume_T2D_r1' (utils/pipeline.py:182), the AGGREGATOR score, not the
# ITM metric the tables report. Measured on MSR-VTT the final model scores 53.4 and the
# aggregator-selected one 52.0 -- so every transfer number we hold was taken from a model
# 1.4 points worse on the reported metric, while GRAM's row uses its authors' final weights.
#
# This is not a thumb on the scale: selecting a checkpoint on MSR-VTT and then reporting
# MSR-VTT would be, which is exactly what we should stop doing. Using the end-of-schedule
# weights is the same protocol GRAM's released checkpoint gets.
#
# MEASURED, and it retires an arm: on the final checkpoint x1_xenc_full_lr2e5 scores 45.7 on
# MSR-VTT against ~54.8 from its selected one, 49.0 on DiDeMo against 51.5, and its video
# features fall to 37.9 against sca's 42.3. Training the whole cross-encoder at 2e-5
# destabilises late, so every win credited to that arm was an early-stopping artifact on a
# checkpoint chosen on MSR-VTT. It is out of the default set.
#
# Also measured: the +1.4 the final checkpoint recovers is MSR-VTT-SPECIFIC, not general --
# transfer moves +0.1, -0.4, +0.2, +0.2. Expected in hindsight, since best_*.pt is selected
# on the MSR-VTT aggregator score and that is the only benchmark where the two differ much.
# The b-arms still need their final-checkpoint numbers: b1's 53.7 came from best_*.pt and is
# not comparable to a table built on final weights.
#
# Separate output root so the existing e1_zs cells and their .done markers stay intact and
# the two can be compared directly.
#
#   sbatch slurm_scripts/e1_final_ckpt.sh
#   SCA_FINAL_ARMS="sca x1_xenc_full_lr2e5" sbatch slurm_scripts/e1_final_ckpt.sh
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
source "$(dirname "$0")/../scripts/cell_done.sh"
mkdir -p slurm_scripts/logs

# the FINAL checkpoint explicitly -- never best_*.pt, which is what this job exists to avoid
final_ckpt() { ls "$1"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1; }

ARMS="${SCA_FINAL_ARMS:-sca b1_bs128_r8 b3_bs512_r8}"
for arm in $ARMS; do
  d="workdir_pretrain/$arm"
  c=$(final_ckpt "$d")
  [ -n "$c" ] && [ -f "$c" ] || { echo "FATAL: no model_step_*.pt under $d/ckpt -- this arm" >&2
    echo "       kept only a best_*.pt, so its final weights are gone and it cannot be" >&2
    echo "       re-evaluated on this protocol." >&2; exit 2; }
  echo "$arm -> $c"
done

cfg_for() { case "$1" in msrvtt) echo "benchmark_eval/configs_depth/sca_msrvtt_tvas.json" ;;
                         *) echo "benchmark_eval/configs_e1/sca_$1.json" ;; esac; }
NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0
for arm in $ARMS; do
  ckpt=$(final_ckpt "workdir_pretrain/$arm")
  for bench in msrvtt didemo activitynet vatex audiocaps; do
    cfg=$(cfg_for "$bench")
    out="workdir/e1_final/${arm}_${bench}"
    cell_is_done "$out" "$cfg" && { echo "== [$arm/$bench] already done, skip"; continue; }
    mkdir -p "$out"
    echo "== [$arm/$bench] START $(date +%T)"
    EVAL_CKPT="$ckpt" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
      --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
      --master_port $((9200 + ${SLURM_JOB_ID:-$$} % 150)) \
      ./benchmark_eval/run_eval.py --config "$cfg" --output_dir "$out" 2>&1 \
      | { grep -v --line-buffered -E "$NOISE" || true; }
    rc=$?
    if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$arm/$bench] OK $(date +%T)"
    else echo "== [$arm/$bench] FAILED rc=$rc" >&2; rc_all=$rc; fi
  done
done
echo
echo "Compare against workdir/e1_zs (the aggregator-selected checkpoint):"
echo "  python3 scripts/raw_vs_itm.py --root workdir/e1_final --pivot"
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
