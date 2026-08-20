#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=e4_transfer
#SBATCH -o ./slurm_scripts/logs/e4_transfer_%j.out
#SBATCH -e ./slurm_scripts/logs/e4_transfer_%j.out
# E4 grids OFF the selection benchmark: sca_t1 / gram / gram_lora on the transfer test
# sets (DiDeMo, ActivityNet, AudioCaps). Decides whether GRAM(-LoRA)'s raw-space edge is
# in-domain-only (E1 says yes for plain GRAM: +3.5 on MSR-VTT flips to -4..-6 off it) --
# the expected picture is SCA above at EVERY rate off-domain. Feature dumps cached
# per (arm, bench); resubmit after a timeout to resume. Grids: 3 mask seeds, 90% incl.
set -uo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
export WANDB_MODE=offline GRAM_MP_CTX=forkserver
mkdir -p results/e4_transfer/feats slurm_scripts/logs

best_ckpt() {
  local b
  b=$(ls -t "$1"/ckpt/best_*.pt 2>/dev/null | head -1)
  [ -n "$b" ] || b=$(ls "$1"/ckpt/model_step_*.pt 2>/dev/null | sort -V | tail -1)
  echo "$b"
}
SCA_T1=$(best_ckpt workdir_pretrain/t1_lr1e4)
SCA_BASE=$(best_ckpt workdir_pretrain/sca)
GRAM=$(best_ckpt workdir_pretrain/gram)
GLORA=$(best_ckpt workdir_pretrain/gram_lora)
for v in SCA_T1 SCA_BASE GRAM GLORA; do [ -f "${!v}" ] || { echo "FATAL: $v ckpt missing" >&2; exit 1; }; done
echo "sca_t1=$SCA_T1"; echo "sca=$SCA_BASE"; echo "gram=$GRAM"; echo "gram_lora=$GLORA"

rc_all=0
for bench in didemo activitynet audiocaps; do
  for arm in sca_t1 sca gram gram_lora; do
    case $arm in
      sca_t1)    ckpt="$SCA_T1";   cfg="benchmark_eval/configs_e1/sca_${bench}.json" ;;
      sca)       ckpt="$SCA_BASE"; cfg="benchmark_eval/configs_e1/sca_${bench}.json" ;;
      gram)      ckpt="$GRAM";   cfg="benchmark_eval/configs_e1/gram_${bench}.json" ;;
      gram_lora) ckpt="$GLORA";  cfg="benchmark_eval/configs_e1/gram_lora_${bench}.json" ;;
    esac
    feats="results/e4_transfer/feats/${arm}_${bench}.pt"
    if [ ! -f "$feats" ]; then
      echo "== [${arm}/${bench}] extracting features"
      EVAL_CKPT="$ckpt" python3 evaluation/run_eval_grids.py --config "$cfg" \
        --dump_features "$feats" || { echo "== [${arm}/${bench}] EXTRACT FAILED" >&2; rc_all=1; continue; }
    fi
    for s in 0 1 2; do
      python3 evaluation/run_eval_grids.py --features "$feats" --seed $s \
        --rates 0 0.25 0.5 0.75 0.9 \
        --out "results/e4_transfer/${arm}_${bench}_s${s}.json" \
        || { echo "== [${arm}/${bench}/s$s] GRID FAILED" >&2; rc_all=1; }
    done
    echo "== [${arm}/${bench}] OK"
  done
done
echo "EXIT=$rc_all DONE $(date +%T)"
exit $rc_all
