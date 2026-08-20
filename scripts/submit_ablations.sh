#!/usr/bin/env bash
# Submit the remaining A1-A9 ablation arms (the ones Wave 4 did not already cover).
#   bash scripts/submit_ablations.sh [--dry] [--a9]
# --a9 includes the five S* -source arms (requires slurm_scripts/build_s_star_a9.sh first).
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=""; A9=""
for a in "$@"; do [ "$a" = --dry ] && DRY=1; [ "$a" = --a9 ] && A9=1; done

ARMS=(
  A1_lmask_term2_off:a1_term2off        # E5: cross-cardinality term ablated
  A3_sstar_identity:a3_sstar_eye        # E6: S* = I (one-hot targets)
  A4_concept_off:a4_concept_off         # E7: L_concept off
  A4_proto_batch:a4_proto_batch         # E7: batch-only prototypes (no EMA)
  A4_eta_0.9:a4_eta09                   # E7: EMA eta 0.99 -> 0.9
  A4_eps_floor_0:a4_eps0                # E7: eps-floor off (collapse check)
  A5_mask_freq:a5_maskfreq              # E4: frequency-weighted m-dagger
  A5_mask_2drop:a5_2drop                # E4: two modalities dropped
  A5_pfull_const_0.5:a5_pconst          # E4: no schedule
  A6_lora_r4:a6_r4                      # E9: LoRA rank 4
  A8_lambda_0:a8_lam0                   # E8: uniformity term off
)
[ -n "$A9" ] && ARMS+=(
  A9_sstar_minilm:a9_minilm  A9_taustar_0.3:a9_tau03  A9_taustar_1.0:a9_tau10
  A9_topk_32:a9_top32        A9_topk_128:a9_top128
)
for spec in "${ARMS[@]}"; do
  cfg="config/sca/ablations/${spec%%:*}.json"; out="workdir_pretrain/${spec##*:}"
  [ -f "$cfg" ] || { echo "MISSING CONFIG: $cfg" >&2; continue; }
  if [ -f "$out/ckpt/best_"*.pt ] 2>/dev/null; then echo "DONE already: $out"; continue; fi
  echo "sbatch slurm_scripts/run_config.sh $cfg $out"
  [ -z "$DRY" ] && sbatch slurm_scripts/run_config.sh "$cfg" "$out"
done
