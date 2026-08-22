#!/usr/bin/env python3
"""Did freezing the reranker's weights recover the lead the reranker was eating?

    python3 scripts/itm_frozen_delta.py

Pairs each cell in workdir/e1_itmfrozen with the SAME cell in workdir/e1_frames -- same arm,
same checkpoint, same benchmark, same eval config except for one key, itm_lora_off.

The validity check comes first and is not optional. itm_lora_off touches stage 2 only: the
cross-encoder's BERT and the condition_feats fed into it. Stage 1, the contrastive scorer
whose R@1 is the AGGREG column, must be untouched. If AGGREG moved between the two roots then
the flag reached stage 1 as well, the two cells differ by more than the reranker, and no
conclusion can be drawn from the ITM column. That is reported as a FAILURE, not a caveat --
a leak this comparison could not see would look exactly like a result.

Then the ITM column, which is the reported metric and the whole question:

  positive  the adapters were degrading the reranker, and the fix is free at inference. The
            recipe follow-up is T14 (config/sca/ablations/T14_itm_frozen.json), which trains
            with the flag on so the ITM branch is fitted on the weights it is scored with.
  ~zero     adapter drift is not what the reranking stage costs. The gap has another cause
            and the frozen-reranker line of attack is closed.
  negative  the adapters were HELPING the reranker despite never seeing an ITM gradient,
            which would be worth understanding before anything else is built on top.

Caveat that bounds what a win here means: these checkpoints were TRAINED with the adapters in
the ITM branch, so evaluating without them is a train/test mismatch by construction. This is a
diagnostic. It licenses running T14; it is not itself a number for a table.
"""
import argparse
import glob
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_vs_itm import BENCHES, scan, split_cell  # noqa: E402

# HyperGRAM's published R@1, the bar the single reported configuration is measured against.
# VATEX is GRAM's released checkpoint measured here instead: HyperGRAM publishes 79.9 against
# our 90.0 on the same weights, a protocol difference too large to compare across.
BAR = {'msrvtt': (56.6, 'HyperGRAM'), 'didemo': (51.3, 'HyperGRAM'),
       'activitynet': (58.2, 'HyperGRAM'), 'vatex': (90.0, 'GRAM ckpt'),
       'audiocaps': (36.1, 'PMRL')}


def collect(root):
    """(arm, benchmark) -> (aggregator R@1, ITM R@1) for every cell under a workdir root."""
    root = root if os.path.isabs(root) else os.path.join(ROOT, root)
    out = {}
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(d):
            continue
        got, _seen = scan(d)
        if not got:
            continue
        arm, bench = split_cell(os.path.basename(d))
        if bench is not None:
            out[(arm, bench)] = (got.get('ret_area_forward'), got.get('ret_itm_area'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', default='workdir/e1_itmfrozen')
    ap.add_argument('--base', default='workdir/e1_frames')
    ap.add_argument('--tol', type=float, default=0.05,
                    help='AGGREG drift tolerated before the pairing is called invalid')
    args = ap.parse_args()

    frozen, base = collect(args.frozen), collect(args.base)
    if not frozen:
        print('no cells under %s -- has itm_frozen_eval.sh finished?' % args.frozen,
              file=sys.stderr)
        print('check the job first:   grep -h "EXIT=" slurm_scripts/logs/itmfrz_*.out',
              file=sys.stderr)
        return 2
    paired = [k for k in frozen if k in base]
    if not paired:
        print('%d frozen cell(s), but none has a partner in %s -- nothing to compare against.'
              % (len(frozen), args.base), file=sys.stderr)
        return 3

    # ---- validity first: stage 1 must be identical, or the ITM column means nothing
    leaked = []
    for k in paired:
        a_f, a_b = frozen[k][0], base[k][0]
        if a_f is not None and a_b is not None and abs(a_f - a_b) > args.tol:
            leaked.append((k, a_b, a_f))
    if leaked:
        print('INVALID: the aggregator score MOVED between the two roots.\n')
        print('%-28s %10s %10s %8s' % ('cell', args.base, 'frozen', 'drift'))
        for (arm, b), a_b, a_f in sorted(leaked):
            print('%-28s %10.1f %10.1f %+8.1f' % ('%s/%s' % (arm, b), a_b, a_f, a_f - a_b))
        print('\nitm_lora_off is meant to touch the reranker only, so stage 1 should be')
        print('bit-identical. It is not, so these cells differ by more than the reranker and')
        print('the ITM column below cannot be attributed to it. Fix that before reading on.')
        return 4

    print('AGGREG identical across %d paired cell(s) (<= %.2f) -- stage 1 untouched, so any'
          % (len(paired), args.tol))
    print('difference below is the reranking stage alone.\n')

    hdr = ('cell', 'AGGREG', 'ITM base', 'ITM frozen', 'DELTA', 'bar', 'vs bar')
    print('%-28s %8s %10s %11s %8s %8s %8s' % hdr)
    print('-' * 86)
    deltas, wins = [], []
    for arm in sorted({a for a, _ in paired}):
        for b in BENCHES:
            if (arm, b) not in frozen or (arm, b) not in base:
                continue
            agg = base[(arm, b)][0]
            i_b, i_f = base[(arm, b)][1], frozen[(arm, b)][1]
            if i_b is None or i_f is None:
                continue
            d = i_f - i_b
            deltas.append(d)
            bar, who = BAR.get(b, (None, ''))
            vs = '%+.1f' % (i_f - bar) if bar is not None else '--'
            if bar is not None and i_f > bar:
                wins.append((b, i_f, bar, who))
            print('%-28s %8s %10.1f %11.1f %+8.1f %8s %8s'
                  % ('%s/%s' % (arm, b), '--' if agg is None else '%.1f' % agg,
                     i_b, i_f, d, '--' if bar is None else '%.1f' % bar, vs))

    if not deltas:
        print('\nno cell had the ITM metric in BOTH roots -- nothing decided.')
        return 3

    mean = sum(deltas) / len(deltas)
    print('\nmean delta over %d cell(s): %+.2f   (min %+.1f, max %+.1f)'
          % (len(deltas), mean, min(deltas), max(deltas)))
    # Six independently trained query-weighted arms span ~1.3 R@1 on the per-benchmark
    # numbers, so a delta inside that band is not distinguishable from run-to-run variation
    # by this comparison alone. Same checkpoint on both sides removes training noise but not
    # the eval-side variation, so the band is an upper bound on what is safely ignorable.
    if mean > 1.0:
        print('\nThe reranker was being degraded by the adapters. This is inference-only, so')
        print('it is free -- but it is a train/test mismatch on these checkpoints. Run T14')
        print('(sbatch --array=16 slurm_scripts/b_grid_pretrain.sh) for the recipe version,')
        print('and report THAT, not this.')
    elif mean < -1.0:
        print('\nThe adapters were HELPING the reranker, despite it never receiving an ITM')
        print('gradient. That is the opposite of the hypothesis and worth understanding')
        print('before building anything else on the two-stage split.')
    else:
        print('\nInside the ~1.3 R@1 spread the six query-weighted arms already show, so this')
        print('does not separate from noise. Adapter drift is not what the reranking stage')
        print('costs, and this line of attack is closed -- do NOT spend a slot on T14.')

    if wins:
        print('\nabove the bar:')
        for b, v, bar, who in wins:
            print('  %-13s %.1f vs %.1f %s (+%.1f)' % (b, v, bar, who, v - bar))
    return 0


if __name__ == '__main__':
    sys.exit(main())
