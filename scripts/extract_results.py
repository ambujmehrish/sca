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

METRIC PROTOCOL (verified against the reference repo's committed eval logs): the numbers
reported by GRAM/HyperAlign tables are the ITM-RERANKED metric (volume_ITM_T2D/D2T in
ret_itm_area) -- their own trained model logs raw volume 25.0 vs ITM 49.0 on msrvtt tvas.
Table rows are therefore emitted from ret_itm_area. ret_area_forward/backard (each
method's raw scorer: centroid/volume/lambda_1) are reported in the summary as the
embedding-space diagnostics that feed E4/E5/E6.
"""
import os
import re
import ast
import json
import glob
import argparse

# two header dialects: training-time '====-evaluation--<name>=====step N--' and the
# run_eval.py runner's '==== evaluation--<name>========' (no step; recorded as step 0)
HEADER = re.compile(r'====[- ]evaluation--(?P<name>.+?)====(?:=step (?P<step>\d+)--)?')
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
            pending = (m.group('name'), int(m.group('step') or 0))
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


def r1_r10(metrics, prefix=None):
    """(R@1, R@10) from a metrics dict; recall strings are 'r1/r5/r10'. prefix selects a
    direction inside multi-direction dicts (e.g. 'volume_ITM_T2D' in ret_itm_area)."""
    keys = [k for k in metrics if prefix is None or k.startswith(prefix)]
    r1 = next((metrics[k] for k in keys if k.endswith('_r1')), None)
    rec = next((metrics[k] for k in keys if k.endswith('_recall')), None)
    r10 = float(rec.split('/')[-1]) if isinstance(rec, str) and rec.count('/') == 2 else None
    return r1, r10


def summarize(family_runs):
    """-> {family[:direction]: {'best': (step, r1, r10), 'final': ..., 'n_evals': int}};
    the itm family is split into :T2D and :D2T pseudo-families."""
    out = {}
    for fam, entries in family_runs.items():
        variants = ([(f'{fam}:T2D', 'volume_ITM_T2D'), (f'{fam}:D2T', 'volume_ITM_D2T')]
                    if 'ret_itm_area' in fam else [(fam, None)])
        for name, prefix in variants:
            scored = [(s, *r1_r10(d, prefix)) for s, d in entries]
            scored = [t for t in scored if t[1] is not None]
            if not scored:
                continue
            out[name] = {'best': max(scored, key=lambda t: t[1]),
                         'final': max(scored, key=lambda t: t[0]),
                         'n_evals': len(scored)}
    return out


def _row_key(family):
    """Table rows come from the ITM-reranked metric (the reference protocol):
    'ret%tvas--msrvtt_ret_ret_itm_area:T2D' -> ('MSR-VTT|T-VAS', 't2v')."""
    m = re.match(r'ret%(\w+)--(\w+?)_ret_ret_itm_area:(T2D|D2T)', family)
    if not m:
        return None, None
    mode = MODE.get(m.group(1))
    bench = BENCH.get(m.group(2))
    if not (mode and bench):
        return None, None
    return f'{bench}|{mode}', ('t2v' if m.group(3) == 'T2D' else 'v2t')


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
        if os.path.isfile(workdir):                      # direct log file (e.g. a slurm .out
            logs = [workdir]                             # section from split_cell_log.py)
        else:
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
            star = ('  <-- TABLE metric (ITM-reranked, reference protocol)'
                    if 'ret_itm_area:T2D' in fam else
                    '  <-- raw scorer (E4/E5/E6 diagnostics)'
                    if 'ret_area_forward' in fam else '')
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
