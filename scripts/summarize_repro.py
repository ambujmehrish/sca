#!/usr/bin/env python3
"""Retrained T9 and seeds against the committed numbers they must reproduce.

    python3 scripts/summarize_repro.py --out experiments/results/harvest/repro_t9.txt

Reference: the committed experiments/results/harvest/raw_vs_itm_frames.txt (read from git HEAD,
so a harvest rewrite on disk cannot change it). New: workdir/e1_frames/<arm>_<bench> eval logs.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_vs_itm import scan  # noqa: E402

REF = 'experiments/results/harvest/raw_vs_itm_frames.txt'
ARMS = ('t9_qweight_only', 's1_t9_seed51', 's2_t9_seed52')
BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')
ROW = re.compile(r'^(\S+)\s+((?:[-+]?\d+\.\d+\s+){5}[-+]?\d+\.\d+)')
FLAG = 1.5          # about one binomial standard error of R@1 on a 1,000-query test set


def reference():
    text = subprocess.run(['git', 'show', 'HEAD:' + REF], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout
    out = {}
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            v = [float(x) for x in m.group(2).split()]
            out[m.group(1)] = {'stage1': v[3], 'itm': v[5]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    ref = reference()
    lines = ['final two-stage T->V R@1: committed -> retrained (delta); CHECK if |delta| > %.1f' % FLAG,
             '%-18s' % 'arm' + ''.join('%22s' % b for b in BENCHES)]
    missing = check = 0
    means = {b: [] for b in BENCHES}
    for arm in ARMS:
        cells = []
        for b in BENCHES:
            r = ref.get('%s_%s' % (arm, b), {}).get('itm')
            got, _ = scan(os.path.join(ROOT, 'workdir/e1_frames', '%s_%s' % (arm, b)))
            n = got.get('ret_itm_area')
            if r is None or n is None:
                cells.append('%22s' % ('%s -> %s' % ('?' if r is None else r, 'MISSING' if n is None else n)))
                missing += 1
                continue
            d = n - r
            means[b].append(n)
            flag = ' CHECK' if abs(d) > FLAG else ''
            check += bool(flag)
            cells.append('%22s' % ('%.1f -> %.1f (%+.1f)%s' % (r, n, d, flag)))
        lines.append('%-18s' % arm + ''.join(cells))
    lines.append('%-18s' % 'retrained mean' + ''.join(
        '%22s' % ('%.1f' % (sum(v) / len(v)) if len(v) == len(ARMS) else 'MISSING') for v in means.values()))
    text = '\n'.join(lines)
    print(text)
    if args.out:
        with open(os.path.join(ROOT, args.out), 'w') as f:
            f.write(text + '\n')
    return 1 if missing or check else 0


if __name__ == '__main__':
    sys.exit(main())
