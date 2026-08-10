#!/usr/bin/env bash
# SCA pre-flight smoke gate (run before ANY cluster hours are spent).
#
# Stage 1  unit + guard suite (CPU, seconds) -- includes LoRA injection against the REAL
#          encoder classes (EVA-CLIP fused-qkv / subln, BEATs, repo BERT) and the GRAM
#          present=None byte-for-byte regression.
# Stage 2  real-data loss-block smoke (CPU, ~2 min): short SCA training on real Flickr8k
#          features (frozen CLIP; downloaded + encoded on first run) with the A10-decided
#          E6 config; hard gates on loss decrease, finite grads, sampler floor, test R@1.
# Stage 3  cluster k=4 smoke (4xGPU, ~30 min): the REAL pipeline end-to-end -- vast27m_150k
#          (T+V,A,S), VAST ckpt, LoRA backbones, S* cache -- 24 steps + validation via
#          config/sca/pretrain_cfg/sca_pretrain_smoke.json. Runs only where $DATA_ROOT and
#          $WORK_ROOT point at the data; otherwise reported loudly as SKIPPED (exit stays 0
#          because stages 1-2 passed; the skip line makes the gap impossible to miss).
#
# Usage:  bash scripts/smoke_test.sh            # stages 1-2 anywhere, +3 on the cluster
#         NPROC=4 bash scripts/smoke_test.sh    # GPUs for stage 3 (default 4)
set -euo pipefail
cd "$(dirname "$0")/.."

# offline compute nodes: use the prefetched HF cache (scripts/prefetch_models.py). When
# MODELS_DIR is set its env.sh MUST exist -- a half-configured offline node should fail
# here, not mid-run.
if [ -n "${MODELS_DIR:-}" ]; then
  if [ -f "$MODELS_DIR/env.sh" ]; then
    source "$MODELS_DIR/env.sh"; echo "[env] offline HF cache: $HF_HOME"
  else
    echo "FATAL: MODELS_DIR=$MODELS_DIR but $MODELS_DIR/env.sh not found -- run scripts/prefetch_models.py on a login node" >&2
    exit 1
  fi
fi

echo "=== [stage 1/3] unit + guard suite ==="
python3 -m pytest tests/ -q

echo "=== [stage 2/3] real-data SCA loss-block smoke (Flickr8k, frozen CLIP) ==="
if [ ! -f experiments/a10_workdir/features_train.pt ]; then
  if [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
    echo "[stage 2] features missing on an OFFLINE node: prefetch with"
    echo "          python3 scripts/prefetch_models.py --models_dir \$WORK_ROOT/sca_models --with-smoke-data"
    echo "          (Flickr8k + CLIP-B/32 then load from the cache; no network needed)"
  else
    echo "[stage 2] features missing -> building from real data (downloads Flickr8k once)"
  fi
  python3 experiments/a10_prepare_flickr8k.py --workdir experiments/a10_workdir
fi
python3 experiments/smoke_sca_losses.py --workdir experiments/a10_workdir

echo "=== [stage 3/3] cluster k=4 pretrain smoke (vast27m_150k, LoRA, 24 steps) ==="
if [ -z "${DATA_ROOT:-}" ] || [ -z "${WORK_ROOT:-}" ]; then
  echo "[stage 3] SKIPPED: export DATA_ROOT and WORK_ROOT to run the real k=4 smoke."
  echo "          This stage is REQUIRED before launching run_pretrain_sca.sh."
  exit 0
fi
if [ ! -f "$DATA_ROOT/vast27m_150k/annotations150k.json" ]; then
  echo "[stage 3] FAILED: $DATA_ROOT/vast27m_150k/annotations150k.json not found" >&2
  exit 1
fi
if [ ! -f "$DATA_ROOT/vast27m_150k/s_star_150k.pt" ]; then
  echo "[stage 3] building the S* cache (one-time preprocessing job)"
  python3 data/semantic_targets.py \
    --annotation_json "$DATA_ROOT/vast27m_150k/annotations150k.json" \
    --out_path "$DATA_ROOT/vast27m_150k/s_star_150k.pt"
fi
NPROC="${NPROC:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export GRAM_MP_CTX="${GRAM_MP_CTX:-forkserver}"
python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node "$NPROC" \
  --master_port 9895 ./run.py --config ./config/sca/pretrain_cfg/sca_pretrain_smoke.json \
  --output_dir ./workdir_smoke_sca --checkpointing true
echo "=== smoke_test.sh: ALL STAGES PASSED ==="
