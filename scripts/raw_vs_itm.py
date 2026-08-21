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
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# extract_results.py already knows where the logs live (workdir/log/log*.txt) and how the
# metric dicts are formatted. Re-deriving that here produced a parser that silently matched
# nothing, so use its functions instead of a second, weaker copy.
from extract_results import parse_log, r1_r10  # noqa: E402

# Family names as parse_log yields them -- the ':T2D' direction split is applied by
# extract_results.summarize(), NOT by the parser, so matching on 'ret_itm_area:T2D' here
# matched nothing and printed the ITM column as '--'. Match the bare family, then select the
# direction with the same prefix summarize() uses.
WANT = ('ret_area_forward', 'ret_itm_area', 'cosine_TV', 'cosine_TA')
ITM_PREFIX = 'volume_ITM_T2D'      # the reported direction inside the ret_itm_area dict


def scan(workdir):
    """metric suffix -> best R@1 over steps, using the extractor's own log parser."""
    out = {}
    for lg in sorted(glob.glob(os.path.join(workdir, 'log', 'log*.txt'))):
        for family, entries in parse_log(lg).items():
            suffix = next((w for w in WANT if family.endswith(w)), None)
            if suffix is None:
                continue
            # ret_itm_area carries both directions in one dict; T2D is the reported one
            prefix = ITM_PREFIX if suffix == 'ret_itm_area' else None
            best = None
            for _step, metrics in entries:
                r1, _r10 = r1_r10(metrics, prefix)
                if r1 is not None and (best is None or r1 > best):
                    best = r1
            if best is not None:
                out[suffix] = best
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
    parsed = 0

    f = lambda v: '--' if v is None else '%.1f' % v
    print('%-34s %8s %8s %8s %8s %8s %8s' %
          ('cell', 'cos T-V', 'cos T-A', 'best 1mod', 'AGGREG', 'TAX', 'ITM'))
    print('-' * 88)
    for d in dirs:
        got = scan(d)
        if not got:
            continue
        parsed += 1
        name = os.path.basename(d)
        tv, ta = got.get('cosine_TV'), got.get('cosine_TA')
        agg, itm = got.get('ret_area_forward'), got.get('ret_itm_area')
        solo = max([v for v in (tv, ta) if v is not None], default=None)
        # aggregation tax: what fusing costs relative to the best single modality it fused.
        # Negative means the aggregator scores WORSE than one of its own inputs.
        tax = '%+.1f' % (agg - solo) if (agg is not None and solo is not None) else '--'
        print('%-34s %8s %8s %8s %8s %8s %8s'
              % (name, f(tv), f(ta), f(solo), f(agg), tax, f(itm)))

    if not parsed:
        # never report an empty comparison as if it were a negative result
        print('\nNO WORKDIR YIELDED METRICS. Expected logs at <workdir>/log/log*.txt --')
        print('check that the eval cells wrote there, and that this is the right --root.')
        return 3
    print('\nTAX is the diagnostic: an aggregator that scores below the best modality it was')
    print('built from is destroying information. Compare an SCA cell against the released-GRAM')
    print('cell for the SAME benchmark -- a worse tax on equal or better single-modality')
    print('features is a fusion defect, not a representation defect.')
    print('\nITM is the reported metric. If SCA leads on AGGREG and trails on ITM, the')
    print('reranking stage is where the advantage is lost instead.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
