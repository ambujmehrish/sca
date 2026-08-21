#!/bin/bash
# Wave 9 -- every baseline retrained at ITS OWN AUTHORS' published recipe, plus the SCA
# ablation grid rebuilt on the reported configuration. Submits everything in parallel.
#
#   bash scripts/submit_recipe_runs.sh --headline    # 4 jobs: confirm Table 1 first
#   bash scripts/submit_recipe_runs.sh --baselines   # + PMRL and HyperGRAM
#   bash scripts/submit_recipe_runs.sh --ablations   # + the 26-arm SCA grid
#   bash scripts/submit_recipe_runs.sh               # everything
#   ... add --dry to print the sbatch lines instead of submitting.
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
DRY=0; WHAT=all; ONLY=
for a in "$@"; do
  case "$a" in
    --dry) DRY=1 ;;
    --headline)  WHAT=headline ;;
    --xenc)      WHAT=xenc ;;
    --baselines) echo "baselines are frozen: GRAM from its released checkpoint + paper, PMRL/HyperGRAM from paper. Nothing to submit." >&2; exit 0 ;;
    --only=*)    ONLY="${a#--only=}" ;;
    --baselines|--baselines-only) WHAT=baselines ;;
    --ablations|--ablations-only) WHAT=ablations ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

mkdir -p slurm_scripts/logs

submit() {  # submit <config> <workdir> [extra args]
  local cfg="$1" out="$2"; shift 2
  local name="${out##*/}"                     # arm name == workdir basename, one identity
  # --only=a,b restricts the submission to named arms. Use it when the account's per-user
  # CPU allowance cannot take the whole phase at once: submit what fits, resubmit the rest
  # as jobs finish. Every arm is independent, so partial submission is always safe.
  if [ -n "$ONLY" ] && ! printf '%s' ",$ONLY," | grep -q ",$name,"; then return 0; fi
  [ -f "$cfg" ] || { echo "FATAL: config $cfg missing" >&2; exit 1; }
  if [ -e "$out" ] && [ ! -f "$out/.provenance" ]; then
    echo "FATAL: $out already exists without a provenance stamp -- refusing to reuse it" >&2
    exit 1
  fi
  # -J/-o/-e override run_config.sh's generic #SBATCH lines, so every job and every log
  # carries the arm name instead of a bare job id: logs/gram_paper_1234567.out, not
  # logs/run_1234567.out. With 30 jobs in flight that is the difference between a readable
  # log directory and thirty anonymous files.
  # --dependency=singleton: at most one job with this name runs at a time, so resubmitting
  # for continuation (a 24h wall clock may still need more than one) queues behind the running job rather
  # than starting a second process that writes the same checkpoints.
  # SCA_CPUS overrides --cpus-per-task when the scheduler rejects the default 32 with
  # "More processors requested than permitted". It is purely a scheduling knob: the
  # dataloader still spawns n_workers per rank, so fewer CPUs means oversubscription and a
  # slower epoch, never a different result. SCA_GPUS exists for symmetry but 4 is the
  # per-node maximum here and the recipe assumes it -- the config's batch_size is GLOBAL
  # (build_dataloader.py:114 divides by world size), so changing it would preserve the
  # recipe but double per-GPU memory. Leave SCA_GPUS at 4.
  # SCA_ACCOUNT / SCA_PARTITION / SCA_QOS override the #SBATCH directives when the default
  # account or partition is unavailable. Scheduling only -- they cannot change what is
  # trained, so a campaign split across accounts stays internally comparable.
  local sched=()
  [ -n "${SCA_ACCOUNT:-}" ]   && sched+=(-A "$SCA_ACCOUNT")
  [ -n "${SCA_PARTITION:-}" ] && sched+=(-p "$SCA_PARTITION")
  [ -n "${SCA_QOS:-}" ]       && sched+=(--qos "$SCA_QOS")
  local cmd=(sbatch -J "$name" --dependency=singleton "${sched[@]}"
             -c "${SCA_CPUS:-32}" --gres="gpu:${SCA_GPUS:-4}"
             -o "slurm_scripts/logs/${name}_%j.out"
             -e "slurm_scripts/logs/${name}_%j.out"
             slurm_scripts/run_config.sh "$cfg" "$out" "$@")
  if [ $DRY -eq 1 ]; then printf '%s\n' "${cmd[*]}"; else "${cmd[@]}"; fi
}

# PHASE 1 -- SCA at the reported recipe. NOTHING here trains a baseline.
# Baselines are frozen as of this revision: GRAM is reported from its released checkpoint
# (evaluated by us) plus its published numbers; PMRL and HyperGRAM from their published
# numbers only, since neither released weights or code. Every GPU hour from here improves
# SCA. The baseline _paper configs are kept for provenance but are no longer submitted.
#
# sca_paper is the one SCA arm never run: lr 1e-4 at batch 128 rather than 256. Batch size
# sets the number of in-batch contrastive negatives, which a centroid objective is directly
# sensitive to, so this can move the number in either direction and is worth knowing.
if [ "$WHAT" = all ] || [ "$WHAT" = headline ]; then
  echo "# --- PHASE 1: SCA at the reported recipe -------------------------------------"
  submit config/sca/pretrain_cfg/sca_paper.json             workdir_pretrain/sca_paper
  submit config/sca/pretrain_cfg/sca_paper_fullft.json      workdir_pretrain/sca_paper_fullft
fi

# PHASE X -- the ITM cross-encoder arms.
# build_optimizer.py freezes 'multimodal_encoder' under the LoRA regime, and that module is
# the ITM cross-encoder (model/sca.py:319 feeds it into self.itm_head). GRAM trains it
# fully -- their L_DAM is one of the two pretraining objectives -- so SCA reranks with a
# rank-8 adapter where GRAM reranks with a finetuned cross-encoder. Measured against the
# released checkpoint SCA leads on every dual-encoder metric (+4.5 DiDeMo, +2.3 ActivityNet
# on the aggregator) and trails after reranking (-0.7, -3.7): a 5-6 point swing at the one
# stage it under-trains, on every benchmark rather than one dataset.
#
# X1 trains the cross-encoder while keeping vision/audio adapted; X2 is the cheaper
# rank-64 version. Both learning rates, because unfreezing this much capacity may prefer
# 2e-5 the way full finetuning did.
if [ "$WHAT" = all ] || [ "$WHAT" = xenc ]; then
  echo "# --- PHASE X: ITM cross-encoder regime ---------------------------------------"
  submit config/sca/ablations/X1_xenc_full.json       workdir_pretrain/x1_xenc_full
  submit config/sca/ablations/X1_xenc_full_lr2e5.json workdir_pretrain/x1_xenc_full_lr2e5
  submit config/sca/ablations/X2_xenc_r64.json        workdir_pretrain/x2_xenc_r64
  submit config/sca/ablations/X2_xenc_r64_lr2e5.json  workdir_pretrain/x2_xenc_r64_lr2e5
fi

if [ "$WHAT" = all ] || [ "$WHAT" = ablations ]; then
  echo "# --- PHASE 3: SCA ablations on the reported config (lr 1e-4, batch 128) -----"
  # Loss-component arms first: they are the Table 6(a) rows and nothing else is blocked on
  # the rest of the grid.
  for arm in A3_sem_off A1_lmask_off A4_concept_off A8_lambda_0 \
             A1_lmask_term2_off A3_sstar_identity \
             A4_proto_batch A4_eta_0.9 A4_eps_floor_0 \
             A5_mask_freq A5_mask_2drop A5_pfull_const_0.5 A5_pfull_end_0.3 \
             A6_lora_r2 A6_lora_r4 A6_lora_r16 A6_lora_r32 A6_lora_r64 \
             A6_lora_asym \
             A7_centroid_gates A8_lambda_0.05 A8_lambda_0.3 A8_unif_weighted; do
    submit "config/sca/ablations_paper/${arm}.json" \
           "workdir_pretrain/abl_$(echo "$arm" | tr 'A-Z.' 'a-z_')"
  done
  # A6_full_ft is deliberately NOT in this list: it resolves to exactly the same config as
  # sca_paper_fullft in Phase 1 (SCA on the paper recipe with use_lora=false). Running both
  # would burn a second full pretrain on an identical experiment -- the ablation table reads
  # its full-finetuning row off the Phase-1 result. scripts/preflight_runs.py enforces this.
  echo "# A6_full_ft: use the Phase-1 sca_paper_fullft result (identical config)"
  echo "# A9_* arms need their S* caches built first (see ablations_paper/MANIFEST.md)"
fi
