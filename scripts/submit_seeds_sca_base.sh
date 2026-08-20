#!/bin/bash
# Seed replicates for the REPORTED SCA configuration (LoRA, lr 2e-5).
#
# The 3-seed error bars in Table 1 were run for the 1e-4 arm (T1_seed51/52) and for GRAM
# (GRAM_seed51/52). Now that 2e-5 is the reported LoRA configuration -- matching every
# baseline's learning rate -- its replicates have to exist too, otherwise our row carries
# no +/- while GRAM's carries +/-0.21.
#
#   bash scripts/submit_seeds_sca_base.sh [--dry]
#
# Seed 42 is the default and is already run (workdir_pretrain/sca). Resubmit each printed
# line until the run reaches num_train_steps; run_config.sh auto-resumes.
set -uo pipefail
cd "$(dirname "$0")/.."
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1

CFG="config/sca/pretrain_cfg/sca_pretrain.json"
[ -f "$CFG" ] || { echo "FATAL: $CFG missing" >&2; exit 1; }

for s in 51 52; do
  cmd=(sbatch slurm_scripts/run_config.sh "$CFG" "workdir_pretrain/sca_seed${s}" --seed "$s")
  if [ $DRY -eq 1 ]; then printf '%s\n' "${cmd[*]}"; else "${cmd[@]}"; fi
done
