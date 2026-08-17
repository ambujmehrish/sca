#!/usr/bin/env python3
"""Extract validation results from pretrain/finetune run logs (Wave-1+ result harvesting).

Parses the trunk's evaluation blocks --

    ====-evaluation--ret%tvas--msrvtt_ret_<family>=====step N--===========
    {'<family>_r1': 44.4, '<family>_recall': '44.4/70.6/80.6', ...}

-- collects every (family, step, metrics), reports the BEST step per family (max R@1 --
matching save_best, so it is the kept checkpoint) alongside the final step, and can emit
measured-row JSONs in the make_latex_tables.py schema.

  python3 scripts/extract_results.py \
      --run "SCA (ours)=workdir_pretrain/sca" \
      --run "GRAM (repro)=workdir_pretrain/gram" \
      --run "PMRL=workdir_pretrain/pmrl" \
      --rows_out results/rows [--section zeroshot_t2v]

The headline families: ret_area_forward is the method's own scorer T->V (centroid for
SCA, volume for GRAM, lambda_1 for PMRL -- each model declares its score_mode), and
ret_area_backard the V->T direction. cosine_* are per-modality diagnostics.
"""
import os
import re
import ast
import json
import glob
import argparse

HEADER = re.compile(r'====-evaluation--(?P<name>.+?)=====step (?P<step>\d+)--')
BENCH = {'msrvtt': 'MSR-VTT', 'didemo': 'DiDeMo', 'activitynet': 'ActivityNet',
         'vatex': 'VATEX', 'audiocaps': 'AudioCaps', 'vggsound': 'VGGSound 5K'}
MODE = {'tv': 'T-V', 'ta': 'T-A', 'tva': 'T-VA', 'tvas': 'T-VAS', 'tvasd': 'T-VASD',
        'tav': 'T-AV'}


def parse_log(path):
    """-> {family: [(step, metrics dict)]}, family like 'ret%tvas--msrvtt_ret_ret_area_forward'."""
    out = {}
    pending = None
    for line in open(path, errors='replace'):
        m = HEADER.search(line)
        if m:
            pending = (m.group('name'), int(m.group('step')))
            continue
        if pending and '{' in line:
            try:
                d = ast.literal_eval(line[line.index('{'): line.rindex('}') + 1])
            except (ValueError, SyntaxError):
                continue
            if isinstance(d, dict):
                out.setdefault(pending[0], []).append((pending[1], d))
                pending = None
    return out


def r1_r10(metrics):
    """(R@1, R@10) from a metrics dict; recall strings are 'r1/r5/r10'."""
    r1 = next((v for k, v in metrics.items() if k.endswith('_r1')), None)
    rec = next((v for k, v in metrics.items() if k.endswith('_recall')), None)
    r10 = float(rec.split('/')[-1]) if isinstance(rec, str) and rec.count('/') == 2 else None
    return r1, r10


def summarize(family_runs):
    """-> {family: {'best': (step, r1, r10), 'final': (step, r1, r10), 'n_evals': int}}"""
    out = {}
    for fam, entries in family_runs.items():
        scored = [(s, *r1_r10(d)) for s, d in entries]
        scored = [t for t in scored if t[1] is not None]
        if not scored:
            continue
        out[fam] = {'best': max(scored, key=lambda t: t[1]),
                    'final': max(scored, key=lambda t: t[0]),
                    'n_evals': len(scored)}
    return out


def _row_key(family):
    """'ret%tvas--msrvtt_ret_ret_area_forward' -> ('MSR-VTT|T-VAS', direction)."""
    m = re.match(r'ret%(\w+)--(\w+?)_ret_ret_area_(forward|backard|backward)', family)
    if not m:
        return None, None
    mode = MODE.get(m.group(1))
    bench = BENCH.get(m.group(2))
    if not (mode and bench):
        return None, None
    direction = 't2v' if m.group(3) == 'forward' else 'v2t'
    return f'{bench}|{mode}', direction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True,
                    metavar='NAME=WORKDIR', help='method name and its output_dir')
    ap.add_argument('--rows_out', help='emit measured-row JSONs here (from BEST steps)')
    ap.add_argument('--section', default='zeroshot_t2v',
                    help='section prefix for the rows: zeroshot|finetune (direction '
                         'suffix _t2v/_v2t is added per family)')
    args = ap.parse_args()

    for spec in args.run:
        name, _, workdir = spec.partition('=')
        logs = sorted(glob.glob(os.path.join(workdir, 'log', 'log*.txt')))
        if not logs:
            print(f'\n== {name}: NO log/log*.txt under {workdir} -- wrong path?')
            continue
        fams = {}
        for lg in logs:                                    # resumed runs append/rotate
            for fam, entries in parse_log(lg).items():
                fams.setdefault(fam, []).extend(entries)
        summary = summarize(fams)
        print(f'\n== {name}  ({workdir}; {len(logs)} log file(s))')
        if not summary:
            print('   no evaluation blocks found (run not validated yet?)')
            continue
        for fam in sorted(summary):
            b, f = summary[fam]['best'], summary[fam]['final']
            star = '  <-- headline' if 'ret_area_forward' in fam else ''
            print(f'   {fam}\n'
                  f'      best  step {b[0]:>6}: R@1 {b[1]:5.1f}  R@10 {b[2] if b[2] is not None else "-"}\n'
                  f'      final step {f[0]:>6}: R@1 {f[1]:5.1f}  R@10 {f[2] if f[2] is not None else "-"}{star}')

        if args.rows_out:
            rows = {'t2v': {}, 'v2t': {}}
            for fam, s in summary.items():
                key, direction = _row_key(fam)
                if key and s['best'][2] is not None:
                    rows[direction][key] = [round(s['best'][1], 1), round(s['best'][2], 1)]
            os.makedirs(args.rows_out, exist_ok=True)
            base = args.section.split('_')[0]
            for direction, r in rows.items():
                if not r:
                    continue
                path = os.path.join(args.rows_out,
                                    f"{name.replace(' ', '_').replace('/', '_')}"
                                    f'_{base}_{direction}.json')
                with open(path, 'w') as fh:
                    json.dump({'method': name, 'section': f'{base}_{direction}',
                               'rows': r}, fh, indent=1)
                print(f'   rows -> {path}')


if __name__ == '__main__':
    main()
