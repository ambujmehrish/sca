# Git-driven job queue

Ported from the FaCT repo's runner. Claude has no cluster access: it submits Slurm jobs on
Leonardo by committing request files and reads results back through git. A runner on a
login node (in tmux) does the cluster side.

```
Claude: ops/queue/<id>.json --commit/push--> branch --git fetch/rebase--> runner (login node)
                                                                           | sbatch / sacct / squeue
Claude <-- experiments/results/ (+ ops/status.json) <-- scripts/ops_collect.sh <-'
```

- **Branch:** `claude/implementation-plan-review-csn6m0`, used by Claude and the runner.
  Override with `OPS_BRANCH`.
- **Account:** `AIFAC_S07_041`, fixed in `scripts/ops_queue.py`; never taken from a request.
- **Limits:** a script is refused if its `#SBATCH` header names another account, more than
  4 GPUs per node, or more than 24:00:00 (4:00:00 on `lrd_all_serial`); a request's `sbatch.time`
  is capped the same way.
- **Slurm scripts:** `slurm_scripts/*.sh` with an `#SBATCH` header, run with the repo as
  working directory.

## First-time setup (login node)

```bash
cd /leonardo_work/AIFAC_S07_041/sca
git status --short | grep -v '^??'          # must print nothing (no local edits to tracked files)
git fetch origin claude/implementation-plan-review-csn6m0
git checkout -B claude/implementation-plan-review-csn6m0 origin/claude/implementation-plan-review-csn6m0
```
The runner and the collector push from this checkout, so `git push` must work here.

## Start / stop

```bash
tmux new -s sca_runner
cd /leonardo_work/AIFAC_S07_041/sca && bash scripts/ops_runner.sh     # Ctrl-b d to detach
```
- **Session name:** `sca_runner`. `fact_runner` is FaCT's and `ops` belongs to another
  project; never touch those.
- **One runner only:** check every login node for an existing sca runner before starting
  one (`pgrep -af ops_runner.sh`). A second runner on the same state file does nothing
  (lock on `<state>.lock`), but a second checkout would submit everything again.
- **Look:** `tmux attach -t sca_runner`, or `tail -f slurm_scripts/logs/ops_runner.log`.
- **Stop:** Ctrl-c inside the session, or `tmux kill-session -t sca_runner`.
- **After a login-node restart** (jobs keep running):
  `tmux new -d -s sca_runner -c /leonardo_work/AIFAC_S07_041/sca "bash -l -c 'bash scripts/ops_runner.sh; exec bash'"`
- `INTERVAL=300` by default. `SCA_ENV_RC` defaults to `/leonardo_work/AIFAC_S07_041/sca_env.rc`;
  `SCA_OPS_BASE` defaults to `$WORK/sca_ops`.

Each loop commits local changes to tracked files under `experiments/results/`, runs
`git fetch origin <branch> && git rebase FETCH_HEAD` (never a merge; on failure
`git rebase --abort` and "git update failed" in the log), then one tick. Any other local edit
to a tracked file, or an untracked file in the way of an incoming one, makes the update fail.

## Where things live

| what | where |
|---|---|
| requests | `ops/queue/<id>.json` (committed by Claude) |
| runner state | `$SCA_OPS_BASE/ops_state/state.json`, outside the repo, so restarts never re-submit |
| status of the queue and of all your jobs | `$SCA_OPS_BASE/results/ops/status.json` -> pushed as `experiments/results/ops/status.json` |
| runner log | `slurm_scripts/logs/ops_runner.log` -> pushed as `experiments/results/logs/ops_runner.txt` |
| Slurm logs | `slurm_scripts/logs/*` changed in the last 7 days -> `experiments/results/logs/<name>.txt` |
| eval / training logs | `workdir*/**/log/log*.txt` and `workdir*/**/run.log` changed in the last 7 days -> `experiments/results/cells/<path>.txt` |
| harvested tables | whatever post scripts write under `experiments/results/` |

Logs over 10 MB keep their first 2,000 and last 20,000 lines. Only json/txt/md/csv files under
20 MB are committed; checkpoints and dumps never are. About once an hour a heartbeat push
sends the status even when no job has finished.

## Request format

One request per file, `ops/queue/<id>.json`; `id` equals the file stem.

```json
{"id": "2026-11-18-rrswap-swap", "what": "reranker swap at 8 frames",
 "script": "rerank_swap.sh",
 "env": {"PHASE": "swap", "GRAM_RELEASED_CKPT": "${WORK}/HyperAlign/pretrained_weights/GRAM_pretrained_TVAS/ckpt/model_step_459.pt"},
 "sbatch": {"time": "24:00:00"},
 "after": ["2026-11-18-rrswap-dump"],
 "post": {"script": "summarize_parity.py", "args": ["--out", "experiments/results/harvest/parity.txt"]},
 "collect": true}
```
Watch-only, for jobs submitted by hand: `{"id", "what", "after_jobs": ["<job id>"], "collect": true}`.

### Rules

- **Keys:** only `id what script env sbatch after after_mode after_jobs post collect`.
- **`id`:** `^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$`, equal to the file stem.
- **`what`:** string of at most 200 characters.
- **`script`:** `^[a-z0-9_]+\.sh$`, existing in `slurm_scripts/`, with a header inside the limits.
- **`env`:** names `^[A-Z][A-Z0-9_]{0,40}$`, not PATH HOME SHELL BASH_ENV ENV IFS PYTHONPATH
  PYTHONHOME PYTHONSTARTUP PROMPT_COMMAND SBATCH_ACCOUNT SLURM_CONF USER LOGNAME and nothing
  starting with `LD_ DYLD_ BASH_FUNC SLURM_ PYTHON`. Values (and post args) match
  `^[A-Za-z0-9_./:+=, %@{}$-]{0,400}$`; `$` only as `${NAME}`/`$NAME` with NAME one of
  `SCA_OPS_BASE CODE_DIR WORK WORK_ROOT FAST SCRATCH_ROOT DATA_ROOT SCA_CACHE_ROOT MODELS_DIR`,
  expanded in Python from the runner's environment (unset is an error). Passed through the
  process environment, never a shell.
- **`sbatch`:** only `time` (`HH:MM:SS`, at most 24:00:00), `qos`, `array`.
- **`after`:** request ids -> `--dependency=afterok:<ids>` (`after_mode: "afterany"` for
  chains that must run even when a part fails or times out).
- **`post.script`:** `^[a-z0-9_]+\.py$` in `scripts/`, run as `python3 scripts/<name> <args>`
  from the repo when every job is terminal (30-minute timeout).
- **`collect`:** true pushes with the message `ops <id>: <what> (<states>)`.

### Cancel / hold

- `ops/cancel/<id>.json` = `{"id", "what", "requests": [request ids]}`: the next tick runs
  `scancel` on those requests' open jobs (only jobs this queue submitted; once each).
- `ops/hold/<id>.json` = `{"id", "what", "action": "hold"|"release", "requests": [...]}`:
  `scontrol hold|release` on those requests' PENDING jobs (once each, in file-name order).

## How Claude uses it

1. Write `ops/queue/<YYYY-MM-DD>-<what>.json`.
2. Run `python3 scripts/ops_queue.py validate --repo .`.
3. Commit and push to the branch; the runner submits within one interval.
4. Read `experiments/results/ops/status.json`, `experiments/results/logs/` and
   `experiments/results/cells/` after `git fetch`. Never rely on pasted terminal output.
