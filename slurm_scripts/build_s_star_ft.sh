#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --job-name=s_star_ft
#SBATCH -o ./slurm_scripts/logs/s_star_ft_%j.out
#SBATCH -e ./slurm_scripts/logs/s_star_ft_%j.out
# Builds the five downstream-finetune S* caches on one GPU (login nodes kill the
# 180k-caption MSR-VTT embed). Rebuilds unconditionally: a cache half-written by a
# killed login-node attempt must not survive. If the dbg queue is full, resubmit with
#   sbatch --qos=normal --time=01:00:00 slurm_scripts/build_s_star_ft.sh
set -euo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
build() {
  echo "== $(date +%T) build $2 from $1"
  python3 data/semantic_targets.py --annotation_json "$1" --out_path "$SCA_CACHE_ROOT/$2"
}
build datasets/annotations/msrvtt/descs_ret_train.json       s_star_ft_msrvtt_train.pt
build datasets/annotations/didemo/descs_ret_train.json       s_star_ft_didemo_train.pt
build datasets/annotations/activitynet/descs_ret_train.json  s_star_ft_activitynet_train.pt
build datasets/annotations/vatex/descs_ret_train_aug.json    s_star_ft_vatex_train_aug.pt
build benchmark_eval/audiocaps_tva_annotation.json           s_star_ft_audiocaps.pt
echo "ALL S* CACHES BUILT $(date +%T):"
ls -la "$SCA_CACHE_ROOT"/s_star_ft_*.pt
