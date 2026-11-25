# Shared by slurm_scripts/frames_parity.sh and slurm_scripts/rerank_swap.sh (sourced, not run).
# Resolves the three checkpoints and runs one eval cell through benchmark_eval/run_eval.py.

T9_ARM="${T9_ARM:-workdir_pretrain/t9_qweight_only}"
PARITY_NOISE="mmco: unref short failure|number of reference frames .+ exceeds max|co located POCs unavailable|UserWarning: The default value of the antialias parameter|^  warnings.warn\($"

# ckpt_for <sca|gram|vast>: the file behind each row; the final step for ours, never best_*
ckpt_for() {
  case "$1" in
    sca)  ls "$T9_ARM"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1 ;;
    gram) echo "${GRAM_RELEASED_CKPT:-}" ;;
    vast) echo "${VAST_CKPT:-$WORK_ROOT/GRAM/code/pretrained_models/VAST_foundation/pretrain_vast/ckpt/model_step_204994.pt}" ;;
    *)    echo "" ;;
  esac
}

# cfg_for <sca|gram|vast> <bench> <fs8|fsbb|tva_fs8|tva_fsbb>: VAST is scored through the GRAM class
cfg_for() {
  local model="$1" bench="$2" fs="$3" cls="$1"
  [ "$cls" = vast ] && cls=gram
  if [ "${fs#tva_}" != "$fs" ]; then echo "benchmark_eval/configs_tva/${cls}_${bench}_${fs#tva_}.json"
  elif [ "$fs" = fsbb ]; then echo "benchmark_eval/configs_fsbb/${cls}_${bench}.json"
  elif [ "$cls" = sca ]; then echo "benchmark_eval/configs_qweight/sca_${bench}.json"
  else echo "benchmark_eval/configs_e1/gram_${bench}.json"; fi
}

# check_ckpt <model>: fail before any GPU time is spent
check_ckpt() {
  local c; c=$(ckpt_for "$1")
  [ -n "$c" ] && [ -f "$c" ] || { echo "FATAL: no checkpoint for $1 (got '$c'; set GRAM_RELEASED_CKPT / VAST_CKPT / T9_ARM)" >&2; return 1; }
  if [ "$1" = sca ]; then
    python3 -c "
import json,sys
m=json.load(open('$T9_ARM/log/hps.json'))['model_cfg']
sys.exit(0 if m.get('sca_query_weighting') and not m.get('sca_frame_slots') and m.get('sca_tau_w',0.1)==0.1 else
         'FATAL: $T9_ARM is not a query-weighted, tau_w=0.1, no-frame-slot arm; configs_qweight would score it wrongly')" || return 1
  fi
  echo "$1 -> $c"
}

# run_cell <out_dir> <cfg> <ckpt> [extra env assignments...]
run_cell() {
  local out="$1" cfg="$2" ckpt="$3"; shift 3
  [ -f "$cfg" ] || { echo "== SKIP: no config $cfg" >&2; return 2; }
  if cell_is_done "$out" "$cfg"; then echo "== [$out] already done, skip"; return 0; fi
  mkdir -p "$out"
  echo "== [$out] START $(date +%T)  cfg=$cfg  ckpt=$ckpt  $*"
  env "$@" EVAL_CKPT="$ckpt" srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 \
    --nproc_per_node "${SLURM_GPUS_ON_NODE:-4}" --master_port $((9200 + ${SLURM_JOB_ID:-$$} % 97)) \
    ./benchmark_eval/run_eval.py --config "$cfg" --output_dir "$out" 2>&1 \
    | { grep -v --line-buffered -E "$PARITY_NOISE" || true; }
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then cell_mark_done "$out" "$cfg"; echo "== [$out] OK $(date +%T)"
  else echo "== [$out] FAILED rc=$rc" >&2; fi
  return "$rc"
}
