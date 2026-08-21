#!/usr/bin/env python3
"""Where in training does each arm actually peak? Read from logs already on disk.

    python3 scripts/training_curve.py
    python3 scripts/training_curve.py --arms sca b1_bs128_r8 --metric itm

Every arm validated valid_freq (=10) times during training, which is roughly twice per
epoch over a 5-epoch run, and each validation logged BOTH the aggregator score and the
ITM-reranked metric. The epoch curve is therefore already recorded for all 45+ arms -- no
new run is needed to ask whether 5 epochs is too many.

Why it matters. GRAM's released checkpoint is model_step_459 on the same VAST foundation we
start from, and reaches 52.5 on MSR-VTT; we spend 2649 steps to reach 53.4. Two collapse
signatures are already visible elsewhere: x1_xenc_full_lr2e5 fell from ~54.8 at its selected
checkpoint to 45.7 at its final one, and sca's aggregator is far worse at the end of training
than at its best step. If arms peak early and decay, every number in the paper was measured
on an overtrained model, and that is a systematic error rather than a tuning opportunity.

Caveat that limits what this can settle: validation during pretraining runs on MSR-VTT only,
so the curve is MSR-VTT's. An arm that peaks early there has not been shown to peak early on
the transfer benchmarks -- it is evidence for running the epoch arms, not a substitute.
"""
import argparse
import glob
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_results import parse_log, r1_r10  # noqa: E402

FAMILY = {'itm': ('ret_itm_area', 'volume_ITM_T2D'),      # the reported metric
          'agg': ('ret_area_forward', None)}              # the aggregator's own score


def curve(workdir, metric):
    """[(step, R@1)] over the run, from the arm's own validation blocks."""
    suffix, prefix = FAMILY[metric]
    points = {}
    for lg in sorted(glob.glob(os.path.join(workdir, 'log', 'log*.txt'))):
        for family, entries in parse_log(lg).items():
            if not family.endswith(suffix):
                continue
            for step, metrics in entries:
                r1, _ = r1_r10(metrics, prefix)
                if r1 is not None:
                    points[step] = r1              # a resumed run may repeat a step
    return sorted(points.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir_root', default='workdir_pretrain')
    ap.add_argument('--arms', nargs='*', help='default: every arm with a log')
    ap.add_argument('--metric', choices=['itm', 'agg'], default='itm')
    args = ap.parse_args()

    root = args.workdir_root if os.path.isabs(args.workdir_root) \
        else os.path.join(ROOT, args.workdir_root)
    dirs = sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d))
    if args.arms:
        want = set(args.arms)
        dirs = [d for d in dirs if os.path.basename(d) in want]
    if not dirs:
        print('no workdirs matched -- run this on the cluster', file=sys.stderr)
        return 2

    print('metric: %s (%s)\n' % (args.metric, FAMILY[args.metric][0]))
    late, early, empty = [], [], []
    for d in dirs:
        name = os.path.basename(d)
        pts = curve(d, args.metric)
        if not pts:
            empty.append(name)
            continue
        last = pts[-1][0]
        best_step, best_v = max(pts, key=lambda t: t[1])
        final_v = pts[-1][1]
        frac = best_step / last if last else 1.0
        print('%-26s %s' % (name, '  '.join('%d:%.1f' % (s, v) for s, v in pts)))
        print('%-26s   peak %.1f at step %d (%.0f%% of the run), final %.1f, drop %+.1f'
              % ('', best_v, best_step, 100 * frac, final_v, final_v - best_v))
        (early if frac <= 0.5 else late).append((name, frac, final_v - best_v))

    print()
    if early:
        print('PEAKED IN THE FIRST HALF -- these decayed for the rest of training:')
        for name, frac, drop in sorted(early, key=lambda t: t[2]):
            print('  %-26s peak at %.0f%% of the run, then %+.1f' % (name, 100 * frac, drop))
        print('\nA reported number from such an arm was measured after the decay. That is a')
        print('systematic error across every affected row, not a tuning opportunity.')
    else:
        print('No arm peaked in the first half: the schedule is not obviously too long on')
        print('MSR-VTT. The epoch arms are still worth running -- validation here is MSR-VTT')
        print('only, and transfer could behave differently -- but the overtraining case is')
        print('not supported by this evidence.')
    if empty:
        print('\n%d arm(s) logged no %s validation: %s'
              % (len(empty), args.metric, ', '.join(empty)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
