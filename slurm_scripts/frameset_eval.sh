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
#SBATCH --job-name=fs_eval
#SBATCH -o ./slurm_scripts/logs/fs_eval_%j.out
#SBATCH -e ./slurm_scripts/logs/fs_eval_%j.out
# Evaluate frame-set arms on all five benchmarks, scored WITH the frame set.
#
# These arms must not go through benchmark_eval/configs_e1: those build one slot per
# modality from the pooled feat_v, so a frame-set checkpoint would be scored as a model that
# was never trained. configs_frames/ sets sca_frame_slots, sca_query_weighting, sca_tau_w and
# dump_frame_feats, and evaluation_mm raises rather than falling back if the flags are on but
# no per-frame features arrive.
#
# The FINAL checkpoint, matching e1_final_ckpt.sh -- never best_*.pt, which save_best selects
# on the aggregator score rather than the reported metric.
#
#   sbatch --dependency=afterok:<t6_jobid> slurm_scripts/frameset_eval.sh
#   SCA_FS_ARMS="t6_frameset t7_frameset_4f" sbatch slurm_scripts/frameset_eval.sh
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
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

final_ckpt() { ls "$1"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1; }

ARMS="${SCA_FS_ARMS:-t6_frameset t7_frameset_4f t8_frameset_tau005 t9_qweight_only t10_frameset_bs256 t11_frameset_tau02}"
FOUND=""
for arm in $ARMS; do
  c=$(final_ckpt "workdir_pretrain/$arm")
  if [ -n "$c" ] && [ -f "$c" ]; then echo "$arm -> $c"; FOUND="$FOUND $arm"
  else echo "SKIP $arm: no model_step_*.pt yet (still training or never ran)"; fi
done
[ -n "$FOUND" ] || { echo "FATAL: no frame-set arm has a final checkpoint" >&2; exit 2; }

NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"
rc_all=0
for arm in $FOUND; do
  ckpt=$(final_ckpt "workdir_pretrain/$arm")
  for bench in msrvtt didemo activitynet vatex audiocaps; do
    # Pick the eval config from what the arm was TRAINED with, read out of its own hps.json --
    # never from its name. Matching on '*qweight*' worked while every query-weighted arm was
    # called t9_qweight_only, and silently broke the moment the sweep arms arrived: g1_r16_qw,
    # s1_t9_seed51 and x3_xenc_clean_lr2e5 are all query-weighted without frames, none of them
    # matches, and all three would have been routed to configs_frames and died on the
    # frame-slots guard. A name is not a record of a configuration; hps.json is.
    #
    #   frame slots      -> configs_frames  (video enters as one slot per frame)
    #   query weighting  -> configs_qweight (weighted centroid over modalities)
    #   neither          -> configs_e1      (uniform centroid)
    #
    # Scoring an arm through the wrong one measures a model that was never trained.
    cfgdir=$(python3 -c "
import json, sys
try:
    m = json.load(open('workdir_pretrain/$arm/log/hps.json'))['model_cfg']
except Exception as e:
    sys.exit('NOHPS %s' % e)
if m.get('sca_frame_slots'):        print('configs_frames')
elif m.get('sca_query_weighting'):  print('configs_qweight')
else:                               print('configs_e1')
" 2>/dev/null)
    case "$cfgdir" in
      configs_frames|configs_qweight|configs_e1) ;;
      *) echo "== [$arm/$bench] SKIP: cannot read workdir_pretrain/$arm/log/hps.json --" >&2
         echo "   refusing to guess the scoring geometry from the arm name." >&2
         rc_all=2; continue ;;
    esac
    cfg="benchmark_eval/${cfgdir}/sca_${bench}.json"
    [ -f "$cfg" ] || { echo "== [$arm/$bench] SKIP: no $cfg" >&2; rc_all=2; continue; }
    out="workdir/e1_frames/${arm}_${bench}"
    cell_is_done "$out" "$cfg" && { echo "== [$arm/$bench] already done, skip"; continue; }
    mkdir -p "$out"
    echo "== [$arm/$bench] START $(date +%T)"
    EVAL_CKPT="$ckpt" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
      --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" \
      --master_port $((9100 + ${SLURM_JOB_ID:-$$} % 90)) \
      ./benchmark_eval/run_eval.py --config "$cfg" --output_dir "$out" 2>&1 \
      | { grep -v --line-buffered -E "$NOISE" || true; }
    rc=$?
    if [ $rc -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$arm/$bench] OK $(date +%T)"
    else echo "== [$arm/$bench] FAILED rc=$rc" >&2; rc_all=$rc; fi
  done
done
echo
echo "  python3 scripts/raw_vs_itm.py --root workdir/e1_frames --pivot"
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
