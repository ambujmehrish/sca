#!/usr/bin/env python3
"""Preflight the campaign submission before any GPU time is spent.

    python3 scripts/preflight_runs.py [--phase headline|baselines|ablations|all]

Drives `scripts/submit_recipe_runs.sh --dry` and checks the plan it prints:

  1. every config exists and parses;
  2. every workdir, job name and log pattern is unique -- two arms sharing any of these
     write over each other's checkpoints or logs;
  3. no two arms resolve to the SAME config -- identical resolved configs mean the same
     experiment is queued twice under two names, which wastes a full pretrain and puts two
     labels on one result. (This check caught sca_paper_fullft == A6_full_ft.)
  4. no workdir already exists without a provenance stamp.

The fingerprint is computed exactly as `slurm_scripts/run_config.sh` computes it -- the
resolved `default` chain plus the trailing CLI args -- so a PASS here means the guard in
that script will also accept every job on first submission.

Exit status 0 = safe to submit.
"""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def fingerprint(cfg, extra):
    """Mirror of run_config.sh's FP: resolved config chain + CLI args."""
    h = hashlib.sha256()

    def feed(p, seen):
        p = p.lstrip('./')
        if p in seen or not os.path.exists(p):
            return
        seen.add(p)
        raw = open(p, 'rb').read()
        h.update(raw)
        try:
            d = json.loads(raw)
        except Exception:
            return
        for sec in ('run_cfg', 'model_cfg'):
            dflt = (d.get(sec) or {}).get('default')
            if dflt:
                feed(dflt, seen)

    feed(cfg, set())
    h.update(('\0'.join(extra)).encode())
    return h.hexdigest()[:16]


def parse_plan(phase):
    cmd = ['bash', 'scripts/submit_recipe_runs.sh', '--dry']
    if phase != 'all':
        cmd.insert(2, f'--{phase}')
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        sys.exit(f'submit_recipe_runs.sh --dry failed:\n{out.stderr}')
    jobs = []
    for line in out.stdout.splitlines():
        if not line.startswith('sbatch'):
            continue
        t = line.split()
        i = t.index('slurm_scripts/run_config.sh')
        jobs.append({'name': t[t.index('-J') + 1], 'log': t[t.index('-o') + 1],
                     'cfg': t[i + 1], 'wd': t[i + 2], 'extra': t[i + 3:]})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='all',
                    choices=['headline', 'baselines', 'ablations', 'all'])
    args = ap.parse_args()
    os.chdir(ROOT)

    jobs = parse_plan(args.phase)
    if not jobs:
        sys.exit('no jobs in the plan -- nothing to check')
    bad = []

    for j in jobs:
        if not os.path.exists(j['cfg']):
            bad.append(f"config missing: {j['cfg']} (arm {j['name']})")
            continue
        try:
            json.load(open(j['cfg']))
        except Exception as e:
            bad.append(f"config unparseable: {j['cfg']}: {e}")

    for label, key in (('workdir', 'wd'), ('job name', 'name'), ('log pattern', 'log')):
        seen = collections.Counter(j[key] for j in jobs)
        dupes = [k for k, v in seen.items() if v > 1]
        print(f'{label:14s} unique {len(seen)}/{len(jobs)}'
              + (f'   DUPLICATED: {dupes}' if dupes else ''))
        if dupes:
            bad.append(f'{label} collision: {dupes}')

    fps = collections.defaultdict(list)
    for j in jobs:
        fps[fingerprint(j['cfg'], j['extra'])].append(j['name'])
    dupes = {k: v for k, v in fps.items() if len(v) > 1}
    print(f'{"resolved cfg":14s} unique {len(fps)}/{len(jobs)}')
    for k, v in dupes.items():
        print(f'    {k}  ->  {v}   SAME EXPERIMENT UNDER TWO NAMES')
        bad.append(f'duplicate experiment: {v}')

    unstamped = [j['wd'] for j in jobs
                 if os.path.exists(j['wd']) and not os.path.exists(f"{j['wd']}/.provenance")]
    print(f'{"workdirs":14s} pre-existing without a stamp: {len(unstamped)}'
          + (f' {unstamped}' if unstamped else ''))
    if unstamped:
        bad.append(f'unstamped existing workdirs: {unstamped}')

    print()
    if bad:
        print(f'PREFLIGHT FAILED ({len(jobs)} jobs)')
        for b in bad:
            print(f'  - {b}')
        return 1
    print(f'PREFLIGHT PASSED -- {len(jobs)} jobs, all distinct, safe to submit')
    return 0


if __name__ == '__main__':
    sys.exit(main())
