#!/bin/bash
# Wait for the GPU partition to have nodes again, then submit a phase once and exit.
#
#   nohup bash scripts/wait_and_submit.sh --headline > slurm_scripts/logs/waiter.log 2>&1 &
#   tail -f slurm_scripts/logs/waiter.log
#
# Written for the state observed on 2026-08-20: `sinfo -p boost_usr_prod` reported
#   boost_usr_prod  up  1-00:00:00  0  n/a
# i.e. the partition was up with ZERO nodes attached, so every sbatch -- any account, any
# size -- was refused with "More processors requested than permitted". Nothing is wrong with
# the job scripts; there is simply nothing to allocate. This poller waits that out.
#
# Env: SCA_PARTITION (default boost_usr_prod), POLL_SECONDS (default 300), plus any of
# SCA_ACCOUNT / SCA_QOS / SCA_CPUS / SCA_GPUS, which are passed through to the submitter.
set -uo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:---headline}"
PART="${SCA_PARTITION:-boost_usr_prod}"
POLL="${POLL_SECONDS:-300}"
MIN_NODES="${MIN_NODES:-1}"

echo "waiter: watching partition '$PART' for >=$MIN_NODES idle/mixed nodes, polling every ${POLL}s"
echo "waiter: will submit '$PHASE' once, then exit"

while true; do
  # %D is the node count per state line; sum the ones that could accept work.
  avail=$(sinfo -h -p "$PART" -o '%D %t' 2>/dev/null \
          | awk '$2=="idle"||$2=="mix"||$2=="alloc" {n+=$1} END {print n+0}')
  total=$(sinfo -h -p "$PART" -o '%D' 2>/dev/null | awk '{n+=$1} END {print n+0}')
  ts=$(date '+%F %T')

  if [ "${total:-0}" -eq 0 ]; then
    echo "$ts  partition has 0 nodes -- still down, waiting"
  elif [ "${avail:-0}" -lt "$MIN_NODES" ]; then
    echo "$ts  $total nodes present but only $avail usable -- waiting"
  else
    echo "$ts  $avail/$total nodes usable -- verifying a job would be accepted"
    if sbatch --test-only ${SCA_ACCOUNT:+-A "$SCA_ACCOUNT"} -p "$PART" \
         ${SCA_QOS:+--qos "$SCA_QOS"} -N1 --ntasks-per-node=1 \
         -c "${SCA_CPUS:-32}" --gres="gpu:${SCA_GPUS:-4}" --mem=240G -t 24:00:00 \
         --wrap 'true' >/dev/null 2>&1; then
      echo "$ts  scheduler accepts the request -- running preflight"
      if python3 scripts/preflight_runs.py --phase "${PHASE#--}"; then
        echo "$ts  preflight passed -- submitting $PHASE"
        bash scripts/submit_recipe_runs.sh "$PHASE"
        echo "$ts  submitted; waiter exiting"
        squeue -u "$USER" -o '%.10i %.20j %.8T %.10M %R'
        exit 0
      fi
      echo "$ts  PREFLIGHT FAILED -- not submitting, waiter exiting so you can look" >&2
      exit 1
    fi
    echo "$ts  nodes present but the scheduler still refuses -- waiting"
  fi
  sleep "$POLL"
done
