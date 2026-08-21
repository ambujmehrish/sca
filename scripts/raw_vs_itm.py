#!/usr/bin/env python3
"""Where does SCA's advantage appear, and where is it lost?

    python3 scripts/raw_vs_itm.py

For each arm and benchmark it reads the eval log and reports two numbers side by side:

  ret_area_forward   the aggregator's OWN score -- centroid for SCA, volume for GRAM.
                     This is what the method contributes.
  ret_itm_area:T2D   the reported metric, after the ITM cross-encoder reranks the top 50.

The gap between them is what the reranking stage does to each method. On DiDeMo and
ActivityNet, SCA leads on the raw scorer (+4.5, +2.3 against the released GRAM checkpoint)
and trails after reranking (-0.7, -3.7) -- a 5-6 point swing at a stage SCA trains with a
rank-8 adapter while GRAM trains the whole cross-encoder.

Whether that pattern is universal decides whether a centroid-native reranker is the right
build or a DiDeMo/ActivityNet curiosity, so this covers every benchmark rather than one.
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
WANT = ('ret_area_forward', 'ret_itm_area:T2D', 'cosine_TV', 'cosine_TA')


def scan(workdir):
    """metric -> best R@1, from whatever eval log the workdir holds."""
    logs = (glob.glob(os.path.join(workdir, '*.txt')) + glob.glob(os.path.join(workdir, 'log/*.txt'))
            + glob.glob(os.path.join(workdir, '*.out')) + glob.glob(os.path.join(workdir, 'log/*')))
    out, cur = {}, None
    for lg in logs:
        if os.path.isdir(lg):
            continue
        try:
            lines = open(lg, errors='replace').read().splitlines()
        except IOError:
            continue
        for ln in lines:
            m = re.search(r'evaluation--[^-]*--\S*?_(ret_area_forward|ret_itm_area:T2D|cosine_TV|cosine_TA)', ln)
            if m:
                cur = m.group(1)
                continue
            if cur:
                r = re.search(r'R@1[:\s]+([0-9.]+)', ln)
                if r:
                    out.setdefault(cur, float(r.group(1)))
                    cur = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='workdir/e1_zs')
    args = ap.parse_args()
    root = args.root if os.path.isabs(args.root) else os.path.join(ROOT, args.root)
    dirs = sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d))
    if not dirs:
        print('no eval workdirs under %s -- run on the cluster' % args.root, file=sys.stderr)
        return 2

    print('%-34s %10s %10s %10s %10s   %s' %
          ('cell', 'cos T-V', 'cos T-A', 'AGGREGATOR', 'ITM', 'ITM - aggregator'))
    print('-' * 96)
    rows = {}
    for d in dirs:
        got = scan(d)
        if not got:
            continue
        name = os.path.basename(d)
        agg = got.get('ret_area_forward')
        itm = got.get('ret_itm_area:T2D')
        rows[name] = (got.get('cosine_TV'), got.get('cosine_TA'), agg, itm)
        f = lambda v: '--' if v is None else '%.1f' % v
        delta = '%+.1f' % (itm - agg) if (agg is not None and itm is not None) else '--'
        print('%-34s %10s %10s %10s %10s   %s' %
              (name, f(rows[name][0]), f(rows[name][1]), f(agg), f(itm), delta))

    print('\nPair an SCA cell with the released-GRAM cell for the same benchmark: if SCA leads')
    print('on AGGREGATOR and trails on ITM, the reranking stage is destroying the advantage')
    print('there too, and the pattern is not specific to DiDeMo/ActivityNet.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
