#!/usr/bin/env python3
"""Did query-weighted reranking move the reported metric, and by how much against HyperGRAM?

    python3 scripts/itm_qw_sweep.py

Reads workdir/e1_itmqw, one cell per (gamma, benchmark), and prints gamma down the rows and
benchmarks across. The gamma = 0 row is a control, not a result: it must reproduce the same
arm's fs_eval number in workdir/e1_frames. If it does not, the extra plumbing perturbed the
baseline and every other row is measuring two changes at once, so that is checked before
anything is reported.

The bar is HyperGRAM's published R@1, except on VATEX where it publishes 79.9 against 90.0 for
the same GRAM weights we measure -- a protocol difference too large to compare across, so the
released GRAM checkpoint is the bar there instead.
"""
import argparse
import glob
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itm_frozen_delta import BAR, collect  # noqa: E402
from raw_vs_itm import BENCHES  # noqa: E402

# gamma=0 must reproduce the baseline, but "reproduce" cannot mean tighter than the benchmark
# reproduces itself. raw_vs_itm.py measures that from cells evaluated twice on one checkpoint:
# 0.0 on MSR-VTT, ActivityNet, VATEX and AudioCaps, and 0.2 on DiDeMo, whose own two runs of
# this exact cell were 51.3 and 51.5.
#
# The first version of this check used a flat 0.05 and duly failed on DiDeMo at -0.2, with the
# other four matching exactly. That is the plumbing being inert and DiDeMo being DiDeMo. A
# per-benchmark tolerance is the honest form: 0.05 is not a claim anyone can make about a cell
# that disagrees with itself by 0.2, and applying it there rejects a valid run.
#
# Both numbers are reported either way, so the reader sees the control drift rather than
# taking a pass on trust.
CONTROL_TOL = {'msrvtt': 0.05, 'didemo': 0.25, 'activitynet': 0.05,
               'vatex': 0.05, 'audiocaps': 0.05}


def split(cell):
    """'t9_qweight_only_g030_didemo' -> ('t9_qweight_only', 'g030', 'didemo')."""
    for b in BENCHES:
        if cell.endswith('_' + b):
            head = cell[:-(len(b) + 1)]
            arm, _, gamma = head.rpartition('_')
            if gamma.startswith('g') and gamma[1:].isdigit():
                return arm, gamma, b
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='workdir/e1_itmqw')
    ap.add_argument('--baseline', default='workdir/e1_frames')
    args = ap.parse_args()

    raw = collect(args.root)          # (arm, bench) -> (aggregator, ITM); arm still holds gamma
    if not raw:
        print('no cells under %s -- has itm_qw_eval.sh finished?' % args.root, file=sys.stderr)
        print('check first:  grep -h "EXIT=" slurm_scripts/logs/itmqw_*.out', file=sys.stderr)
        return 2
    base = collect(args.baseline)

    cells, gammas, arms = {}, set(), set()
    for (armgamma, bench), (_agg, itm) in raw.items():
        arm, gamma, b = split('%s_%s' % (armgamma, bench))
        if b is None or itm is None:
            continue
        cells[(arm, gamma, b)] = itm
        gammas.add(gamma); arms.add(arm)
    if not cells:
        print('cells found but none parsed as <arm>_<gamma>_<bench>', file=sys.stderr)
        return 3

    for arm in sorted(arms):
        # ---- control first. gamma=0 adds no cross-encoder passes and must be the old number.
        ctrl = {b: cells.get((arm, 'g000', b)) for b in BENCHES}
        ref = {b: base.get((arm, b), (None, None))[1] for b in BENCHES}
        drift = [(b, ref[b], ctrl[b], ctrl[b] - ref[b]) for b in BENCHES
                 if ctrl[b] is not None and ref[b] is not None]
        bad = [d for d in drift if abs(d[3]) > CONTROL_TOL[d[0]]]
        print('\n=== %s' % arm)
        if drift:
            print('control (gamma=0 vs %s), tolerance = that cell\'s own reproducibility:'
                  % args.baseline)
            for b, r, c, d in drift:
                print('  %-13s baseline %.1f  gamma=0 %.1f  %+.1f   (tol %.2f)%s'
                      % (b, r, c, d, CONTROL_TOL[b], '  <- OVER' if abs(d) > CONTROL_TOL[b] else ''))
        if bad:
            print('\nCONTROL FAILED on %d benchmark(s). gamma=0 adds no cross-encoder pass, so'
                  % len(bad))
            print('it must be the untouched protocol. Where it is not, the plumbing moved the')
            print('baseline and those rows would measure two changes at once.')
            continue
        if not any(ctrl.values()):
            print('no gamma=0 cell yet -- run it before trusting any other row:')
            print('  sbatch --array=0-4 slurm_scripts/itm_qw_eval.sh')
        else:
            print('control OK: every benchmark within its own reproducibility.')

        print('\n%-8s %s %8s' % ('gamma', ' '.join('%12s' % b for b in BENCHES), 'mean'))
        print('-' * (8 + 13 * len(BENCHES) + 9))
        rows = {}
        for g in sorted(gammas):
            vals = [cells.get((arm, g, b)) for b in BENCHES]
            got = [v for v in vals if v is not None]
            mean = sum(got) / len(got) if got else None
            rows[g] = (vals, mean)
            print('%-8s %s %8s'
                  % (g, ' '.join('%12s' % ('--' if v is None else '%.1f' % v) for v in vals),
                     '--' if mean is None else '%.2f' % mean))

        print('\n%-8s %s' % ('vs bar', ' '.join('%12s' % b for b in BENCHES)))
        print('%-8s %s' % ('', ' '.join('%12.1f' % BAR[b][0] for b in BENCHES)))
        best_g = max((g for g in rows if rows[g][1] is not None),
                     key=lambda g: rows[g][1], default=None)
        for g in sorted(gammas):
            vals, _ = rows[g]
            marks = []
            for b, v in zip(BENCHES, vals):
                marks.append('--' if v is None else '%+.1f' % (v - BAR[b][0]))
            print('%-8s %s' % (g, ' '.join('%12s' % m for m in marks)))

        if best_g is None:
            continue
        base_mean = rows.get('g000', (None, None))[1]
        gain = rows[best_g][1] - base_mean if base_mean is not None else None
        print('\nbest gamma by mean: %s' % best_g, end='')
        if gain is not None:
            print('  (%+.2f R@1 mean over gamma=0)' % gain)
        else:
            print()
        wins = [(b, cells[(arm, best_g, b)]) for b in BENCHES
                if cells.get((arm, best_g, b)) is not None
                and cells[(arm, best_g, b)] > BAR[b][0]]
        if wins:
            print('above the bar at %s:' % best_g)
            for b, v in wins:
                print('  %-13s %.1f vs %.1f %s (+%.1f)' % (b, v, BAR[b][0], BAR[b][1],
                                                           v - BAR[b][0]))
        else:
            print('no benchmark clears its bar at any gamma.')
        if gain is not None and abs(gain) < 0.2:
            print('\nThe whole sweep moves less than the 0.2 R@1 eval floor. Reranking is not')
            print('reachable this way, and gamma should not appear in the paper as a knob.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
