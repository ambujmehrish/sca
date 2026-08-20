#!/bin/bash
# Wave 9 -- every baseline retrained at ITS OWN AUTHORS' published recipe, plus the SCA
# ablation grid rebuilt on the reported configuration. Submits everything in parallel.
#
#   bash scripts/submit_recipe_runs.sh [--dry] [--baselines-only|--ablations-only]
#
# WHY: our first-pass reproductions all trained at lr 2e-5 with batch 256, inherited from
# the HyperAlign lineage. GRAM's released config pretrains at lr 1e-4 / batch 128 and PMRL's
# paper at lr 2e-5 / batch 64, so neither reproduction matched its source. Separately, all
# 26 A1-A9 ablation arms were deltas on the lr-2e-5 SCA config while the reported SCA is the
# lr-1e-4 one, so they ablated a model that is not in the paper. And the existing 1e-4 SCA
# arm used batch 256 against GRAM's 128, so even it was not matched. See RECIPE_AUDIT.md.
#
# NAMING (one scheme, no collisions with the first-pass runs):
#   config  config/{baselines,sca}/pretrain_cfg/<method>_paper.json  -- GRAM's recipe
#           config/sca/ablations_paper/<ARM>.json                     -- ablations on it
#   workdir workdir_pretrain/<method>_paper       -- never reuses a first-pass dir
#           workdir_pretrain/abl_<arm>
# The first-pass workdirs (gram, pmrl, gram_lora, sca, t1_lr1e4, a6_*, a7_*, a8_*) are left
# untouched, so both generations stay separately extractable and nothing is overwritten.
#
# run_config.sh stamps each workdir with a fingerprint of its resolved config chain and
# refuses to resume when the stamp disagrees, so a mistyped output_dir fails loudly instead
# of silently continuing another arm's weights.
set -uo pipefail
cd "$(dirname "$0")/.."
DRY=0; WHAT=all
for a in "$@"; do
  case "$a" in
    --dry) DRY=1 ;;
    --baselines-only) WHAT=baselines ;;
    --ablations-only) WHAT=ablations ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

submit() {  # submit <config> <workdir> [extra args]
  local cfg="$1" out="$2"; shift 2
  [ -f "$cfg" ] || { echo "FATAL: config $cfg missing" >&2; exit 1; }
  if [ -e "$out" ] && [ ! -f "$out/.provenance" ]; then
    echo "FATAL: $out already exists without a provenance stamp -- refusing to reuse it" >&2
    exit 1
  fi
  local cmd=(sbatch slurm_scripts/run_config.sh "$cfg" "$out" "$@")
  if [ $DRY -eq 1 ]; then printf '%s\n' "${cmd[*]}"; else "${cmd[@]}"; fi
}

if [ "$WHAT" = all ] || [ "$WHAT" = baselines ]; then
  echo "# --- baselines at their authors' recipes -------------------------------------"
  submit config/baselines/pretrain_cfg/gram_paper.json      workdir_pretrain/gram_paper
  submit config/baselines/pretrain_cfg/pmrl_paper.json      workdir_pretrain/pmrl_paper
  submit config/baselines/pretrain_cfg/gram_lora_paper.json workdir_pretrain/gram_lora_paper
  submit config/baselines/pretrain_cfg/gram_hyp_paper.json  workdir_pretrain/gram_hyp_paper
  echo "# --- SCA under the SAME recipe (lr 1e-4, batch 128) --------------------------"
  # The existing 1e-4 SCA arm (t1_lr1e4) trained at batch 256, so it is NOT matched to
  # gram_paper: batch size sets the number of contrastive negatives, which a centroid
  # objective is directly sensitive to. These two rows are the real head-to-head.
  submit config/sca/pretrain_cfg/sca_paper.json         workdir_pretrain/sca_paper
  submit config/sca/pretrain_cfg/sca_paper_fullft.json  workdir_pretrain/sca_paper_fullft
fi

if [ "$WHAT" = all ] || [ "$WHAT" = ablations ]; then
  echo "# --- SCA ablations on the reported configuration (lr 1e-4, batch 128) -------"
  # Loss-component arms first: they are the Table 6(a) rows and nothing else is blocked on
  # the rest of the grid.
  for arm in A3_sem_off A1_lmask_off A4_concept_off A8_lambda_0 \
             A1_lmask_term2_off A3_sstar_identity \
             A4_proto_batch A4_eta_0.9 A4_eps_floor_0 \
             A5_mask_freq A5_mask_2drop A5_pfull_const_0.5 A5_pfull_end_0.3 \
             A6_lora_r4 A6_lora_r16 A6_lora_asym A6_full_ft \
             A7_centroid_gates A8_lambda_0.05 A8_lambda_0.3 A8_unif_weighted; do
    submit "config/sca/ablations_paper/${arm}.json" \
           "workdir_pretrain/abl_$(echo "$arm" | tr 'A-Z.' 'a-z_')"
  done
  echo "# A9_* arms need their S* caches built first (see ablations_paper/MANIFEST.md)"
fi
