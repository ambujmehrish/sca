#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --job-name=s_star_a9
#SBATCH -o ./slurm_scripts/logs/s_star_a9_%j.out
#SBATCH -e ./slurm_scripts/logs/s_star_a9_%j.out
# A9 (S* source sensitivity) needs five alternative S* caches over the SAME 150k
# annotations. Each is a full sentence-encoder pass -> GPU, not the login node.
set -euo pipefail
source "${SCA_ENV_RC:-/leonardo_work/AIFAC_S07_041/sca_env.rc}"
cd "$CODE_DIR"
[ -f "$MODELS_DIR/env.sh" ] && source "$MODELS_DIR/env.sh" || { echo "FATAL: prefetch first" >&2; exit 1; }
ANN="$DATA_ROOT/vast27m_150k/annotations150k.json"
build() {  # <out> <extra args...>
  local out="$SCA_CACHE_ROOT/$1"; shift
  [ -f "$out" ] && { echo "== $out exists, skip"; return; }
  echo "== $(date +%T) building $out  ($*)"
  python3 data/semantic_targets.py --annotation_json "$ANN" --out_path "$out" "$@"
}
build s_star_150k_minilm.pt --model_name sentence-transformers/all-MiniLM-L6-v2
build s_star_150k_tau03.pt  --tau_star 0.3
build s_star_150k_tau10.pt  --tau_star 1.0
build s_star_150k_top32.pt  --topk 32
build s_star_150k_top128.pt --topk 128
echo "ALL A9 CACHES BUILT $(date +%T)"; ls -la "$SCA_CACHE_ROOT"/s_star_150k_*.pt
