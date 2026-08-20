#!/usr/bin/env python3
"""Verify, after the fact, that each run trained with the config we intended.

    python3 scripts/verify_runs.py [--phase headline|baselines|ablations|all]

`utils/args.py` dumps the fully resolved options to `<workdir>/log/hps.json` at the start of
every run. That file is ground truth for what the job actually used -- it is written after
the `default` chain is merged and after any CLI override. This script compares it against
the config the submitter intended for that workdir, and against the `.provenance` stamp.

Preflight (scripts/preflight_runs.py) proves the *plan* is sound before submission; this
proves the *outcome* matches the plan afterwards. Run it once the first checkpoints appear
and again before extracting results into tables.

Statuses: OK, MISMATCH (trained with something other than intended -- do not use its
numbers), NOT STARTED, NO STAMP.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preflight_runs import ROOT, fingerprint, parse_plan   # noqa: E402

# (label, where to read it from the resolved dump, where from the source config)
FIELDS = [
    ('lr',    lambda h: h['run_cfg'].get('learning_rate'),
              lambda c: c['run_cfg'].get('learning_rate')),
    ('batch', lambda h: (h['data_cfg']['train'] or [{}])[0].get('batch_size'),
              lambda c: (c['data_cfg']['train'] or [{}])[0].get('batch_size')),
    ('model', lambda h: h['model_cfg'].get('model_type'),
              lambda c: c['model_cfg'].get('model_type')),
    ('lora',  lambda h: h['model_cfg'].get('use_lora'),
              lambda c: c['model_cfg'].get('use_lora')),
    ('rank',  lambda h: h['model_cfg'].get('lora_r_vision'),
              lambda c: c['model_cfg'].get('lora_r_vision')),
    ('alpha', lambda h: h['model_cfg'].get('lora_alpha'),
              lambda c: c['model_cfg'].get('lora_alpha')),
]


def resolve(path, seen=None):
    """Merge a config's `default` chain the way run.py does."""
    seen = seen or set()
    if path in seen:
        return {}
    seen.add(path)
    d = json.load(open(path))
    out = {}
    for sec in ('run_cfg', 'model_cfg'):
        s = dict(d.get(sec, {}))
        dflt = s.pop('default', None)
        base = {}
        # NOT lstrip('./'): it strips characters, not a prefix, and mangles absolute paths
        dflt_path = dflt[2:] if dflt and dflt.startswith('./') else dflt
        if dflt_path and os.path.exists(dflt_path):
            base = json.load(open(dflt_path))
            base.pop('default', None)
        merged = dict(base)
        merged.update(s)
        out[sec] = merged
    out['data_cfg'] = d.get('data_cfg', {})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='all',
                    choices=['headline', 'baselines', 'ablations', 'all'])
    args = ap.parse_args()
    os.chdir(ROOT)

    jobs = parse_plan(args.phase)
    rows, bad = [], 0
    for j in jobs:
        wd, cfg = j['wd'], j['cfg']
        hps_path = os.path.join(wd, 'log', 'hps.json')
        if not os.path.exists(hps_path):
            rows.append((j['name'], 'NOT STARTED', ''))
            continue

        hps = json.load(open(hps_path))
        want = resolve(cfg)
        diffs = []
        for label, from_hps, from_cfg in FIELDS:
            try:
                got = from_hps(hps)
            except Exception:
                got = '<unreadable>'
            exp = from_cfg(want)
            if exp is not None and got != exp:
                diffs.append(f'{label}: ran {got!r}, config says {exp!r}')

        stamp = os.path.join(wd, '.provenance')
        if not os.path.exists(stamp):
            diffs.append('no .provenance stamp')
        else:
            have = ''
            for line in open(stamp):
                if line.startswith('fingerprint='):
                    have = line.split('=', 1)[1].strip()
            if have != fingerprint(cfg, j['extra']):
                diffs.append(f'stamp fingerprint {have} != config fingerprint '
                             f'{fingerprint(cfg, j["extra"])}')

        if diffs:
            bad += 1
            rows.append((j['name'], 'MISMATCH', '; '.join(diffs)))
        else:
            lr = hps['run_cfg'].get('learning_rate')
            bs = (hps['data_cfg']['train'] or [{}])[0].get('batch_size')
            rows.append((j['name'], 'OK', f'lr={lr} batch={bs}'))

    w = max(len(r[0]) for r in rows) if rows else 10
    for name, status, note in rows:
        print(f'{name:{w}s}  {status:11s}  {note}')
    started = sum(1 for r in rows if r[1] != 'NOT STARTED')
    print(f'\n{started}/{len(rows)} started, {bad} mismatched')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
