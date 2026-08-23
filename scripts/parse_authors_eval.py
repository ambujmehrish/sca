#!/usr/bin/env python3
"""Read retrieval numbers out of the authors' own eval logs.

    python3 scripts/parse_authors_eval.py workdir/pmrl_released/*/run.log
    python3 scripts/parse_authors_eval.py --json workdir/pmrl_released/*/run.log

Both released repositories print evaluation through the same VAST-family logger:

    ==== evaluation--ret%tvas--vatex_ret_vatex_ret_ret_itm_tvas========
    {'video_r1': 89.6, 'video_recall': '89.6/97.9/98.8', 'video_ravg': 95.4, 'txt_r1': 87.0, ...}

so one parser serves the PMRL released-checkpoint row and the HyperGRAM authors'-code row.

WHICH NUMBER IS THE ROW. `ret_itm_*` -- the two-stage figure after ITM reranking -- because
that is what every other row of our table reports and what these papers report. `ret_itc_*`
is the aggregator before reranking: useful for the aggregation-tax analysis, never the
headline. Reporting the wrong one of those two would move a row by tens of points, so both
are extracted and labelled rather than one being picked silently.

`video_r1` is text->video (retrieving the video for a caption); `txt_r1` is the reverse. Our
table reports text->video throughout.
"""
import argparse
import ast
import json
import os
import re
import sys

HEADER = re.compile(r'====\s*evaluation--(?P<task>[^-]+)--(?P<rest>\S*?)_(?P<metric>'
                    r'ret_itm_\w+|ret_itc_\w+|cosine_[A-Z]{2}|eigenvalue_max|'
                    r'eigenvector_uniformity)========')
BENCH = re.compile(r'([a-z0-9]+)_ret')


def parse(path):
    """Every metric block in one log, latest occurrence winning."""
    out = {}
    lines = open(path, errors='replace').read().splitlines()
    for i, line in enumerate(lines):
        m = HEADER.search(line)
        if not m:
            continue
        # the payload dict is on one of the next couple of lines, after the logger prefix
        for j in range(i + 1, min(i + 4, len(lines))):
            brace = lines[j].find('{')
            if brace < 0:
                continue
            try:
                payload = ast.literal_eval(lines[j][brace:].strip())
            except (ValueError, SyntaxError):
                continue
            b = BENCH.search(m.group('rest'))
            out[(b.group(1) if b else '?', m.group('metric'))] = payload
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--json', action='store_true', help='machine-readable, for the table builder')
    args = ap.parse_args()

    everything = {}
    for path in args.logs:
        if not os.path.exists(path):
            sys.exit('FATAL: %s not found' % path)
        got = parse(path)
        if not got:
            print('%s: NO EVALUATION BLOCKS -- the run did not reach evaluation' % path,
                  file=sys.stderr)
            continue
        for (bench, metric), payload in got.items():
            everything.setdefault(bench, {})[metric] = payload

    if args.json:
        print(json.dumps(everything, indent=1, sort_keys=True))
        return 0

    for bench in sorted(everything):
        met = everything[bench]
        itm = next((v for k, v in met.items() if k.startswith('ret_itm')), None)
        itc = next((v for k, v in met.items() if k.startswith('ret_itc')), None)
        print('\n%s' % bench.upper())
        if itm:
            print('  REPORTED  ret_itm  T->V R@1 %5.1f   recall %s'
                  % (itm.get('video_r1', float('nan')), itm.get('video_recall', '?')))
            if 'txt_r1' in itm:
                print('            (V->T R@1 %.1f, not the reported direction)' % itm['txt_r1'])
        else:
            print('  REPORTED  ret_itm  MISSING -- no reranked block in this log')
        if itc:
            print('  aggregator ret_itc T->V R@1 %5.1f' % itc.get('video_r1', float('nan')))
        # the aggregation tax: the aggregator against the best single pathway it is built from
        singles = {k: v.get('forward_r1') for k, v in met.items() if k.startswith('cosine_')
                   and v.get('forward_r1') is not None}
        if itc and singles:
            best = max(singles.items(), key=lambda kv: kv[1])
            print('  best single pathway %s %.1f  ->  aggregation tax %+.1f'
                  % (best[0], best[1], itc.get('video_r1', 0) - best[1]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
