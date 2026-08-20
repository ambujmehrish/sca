#!/bin/bash
# Recipe-fairness runs: the baselines at lr 1e-4.
#
# Every baseline we trained (GRAM, GRAM-LoRA, PMRL, HyperGRAM) used lr 2e-5, inherited
# from the HyperAlign lineage. SCA has BOTH 2e-5 and 1e-4 arms. So any table that reports
# SCA at 1e-4 against a baseline at 2e-5 is comparing a tuned method against an untuned
# one -- and GRAM's own paper trains at 1e-4, so our GRAM reproduction is under-tuned
# relative to its own publication. These runs close that gap: after they land, both
# learning rates exist for both families and whichever is reported is matched.
#
#   bash scripts/submit_baselines_lr1e4.sh [--dry]
#
# Each is a full 150k-clip pretrain -- resubmit the printed sbatch line until the run
# reaches num_train_steps (run_config.sh auto-resumes from its own checkpoints).
set -uo pipefail
cd "$(dirname "$0")/.."
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1

ARMS=(
  gram_pretrain_lr1e4:gram_lr1e4              # the row that decides the headline margin
  pmrl_pretrain_lr1e4:pmrl_lr1e4
  gram_lora_pretrain_lr1e4:gram_lora_lr1e4    # LoRA-vs-LoRA at a matched lr
)

for arm in "${ARMS[@]}"; do
  cfg="config/baselines/pretrain_cfg/${arm%%:*}.json"
  out="workdir_pretrain/${arm##*:}"
  [ -f "$cfg" ] || { echo "FATAL: $cfg missing" >&2; exit 1; }
  cmd=(sbatch slurm_scripts/run_config.sh "$cfg" "$out")
  if [ $DRY -eq 1 ]; then printf '%s\n' "${cmd[*]}"; else "${cmd[@]}"; fi
done
