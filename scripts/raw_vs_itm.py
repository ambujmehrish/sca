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
    """(metric suffix -> R@1 from the LAST run, suffix -> EVERY run's R@1 in log order).

    The full list, not just a count. Where a cell was evaluated more than once the repeats are
    the SAME checkpoint scored by the SAME config, so the spread between them is the eval
    noise floor -- measured, on data already on disk, rather than assumed. Any margin in the
    paper smaller than that spread is not a result.
    """
    out, seen = {}, {}
    for lg in sorted(glob.glob(os.path.join(workdir, 'log', 'log*.txt'))):
        for family, entries in parse_log(lg).items():
            suffix = next((w for w in WANT if family.endswith(w)), None)
            if suffix is None:
                continue
            # ret_itm_area carries both directions in one dict; T2D is the reported one
            prefix = ITM_PREFIX if suffix == 'ret_itm_area' else None
            for _step, metrics in entries:
                r1, _r10 = r1_r10(metrics, prefix)
                if r1 is not None:
                    # LAST, not max. These are EVAL workdirs: one entry per run of the cell.
                    # Taking the max was max-over-REPEATS -- a cell re-evaluated twice (as the
                    # frame-set cells were, when fs_eval ran a second time to pick up arms that
                    # had not finished) would silently report the luckier of the two runs.
                    # Max over steps is right for a training log and wrong here.
                    out[suffix] = r1
                    seen.setdefault(suffix, []).append(r1)
    return out, seen


BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')


def split_cell(name):
    """'sca_didemo_t1' -> ('sca_t1', 'didemo'); 'x1_xenc_full_vatex' -> ('x1_xenc_full', 'vatex').

    The benchmark is a known token, and whatever surrounds it is the arm -- cell names put a
    seed/variant suffix AFTER the benchmark, so it has to be rejoined rather than dropped.
    """
    for b in BENCHES:
        for pat in ('_%s_' % b, '_%s' % b):
            if name.endswith(pat.rstrip('_')) or pat in name:
                head, _, tail = name.partition('_%s' % b)
                arm = head + tail          # tail keeps a leading '_' when a suffix follows
                return (arm or name), b
    return name, None


def pivot(rows, metric_idx, title, out=sys.stdout):
    """arm x benchmark table for one metric, so arms can be compared down a column."""
    cells, arms = {}, []
    for name, vals in rows.items():
        arm, bench = split_cell(name)
        if bench is None:
            continue
        cells[(arm, bench)] = vals[metric_idx]
        if arm not in arms:
            arms.append(arm)
    if not cells:
        return
    present = [b for b in BENCHES if any((a, b) in cells for a in arms)]
    print('\n%s' % title, file=out)
    print('%-24s %s' % ('arm', ' '.join('%12s' % b for b in present)), file=out)
    print('-' * (24 + 13 * len(present)), file=out)
    for arm in sorted(arms):
        line = ' '.join('%12s' % ('--' if cells.get((arm, b)) is None else
                                  '%.1f' % cells[(arm, b)]) for b in present)
        print('%-24s %s' % (arm, line), file=out)


# The unimodal cosine scores are the encoder outputs with no fusion on top. Two runs of one
# cell that agree on the aggregator but disagree HERE cannot be two runs of one model.
ENCODER_METRICS = ('cosine_TV', 'cosine_TA')


def noise_floor(repeats, out=sys.stdout):
    """What a repeated evaluation of the SAME checkpoint by the SAME config disagrees by.

    Some cells were evaluated twice: fs_eval was rerun to pick up arms that had not finished
    the first time, and it re-scored the ones that had. Where the checkpoint was the same both
    times those repeats are free replicates -- same weights, same data, same config -- so
    everything separating them is eval-side nondeterminism (fp16 reduction order, sharding
    across ranks, decode jitter on the video reader).

    NOT ALL OF THEM ARE. fs_eval resolves the checkpoint with `ls model_step_*.pt | tail -1`,
    so an arm that trained further between the two runs was scored at a LATER step the second
    time. Those two numbers differ because the model differs, and folding them into a noise
    floor inflates it -- which would then excuse real margins as noise, the most expensive
    direction to be wrong in.

    They are separable without any extra run. cosine_TV and cosine_TA are raw encoder outputs
    with no fusion above them: identical weights give identical values, deterministically. A
    cell whose unimodal scores moved was scored on two different checkpoints, and is reported
    separately rather than counted.

    The surviving number decides how the main table may be read. A reported margin below it is
    not a measurement of anything, however carefully it was extracted."""
    if not repeats:
        return
    # repeats: cell -> {metric suffix: [every run's R@1]}
    same, moved = {}, {}
    for cell, metrics in repeats.items():
        drift = max((max(v) - min(v) for k, v in metrics.items()
                     if k in ENCODER_METRICS and len(v) > 1), default=0.0)
        (moved if drift > 0.05 else same)[cell] = (metrics, drift)

    print('\nEVAL NOISE FLOOR -- repeated runs of one cell on the SAME checkpoint', file=out)
    print('%-34s %-18s %20s %8s' % ('cell', 'metric', 'runs', 'spread'), file=out)
    print('-' * 84, file=out)
    worst, worst_cell = 0.0, None
    for cell in sorted(same):
        for suffix, vals in sorted(same[cell][0].items()):
            if len(vals) < 2:
                continue
            spread = max(vals) - min(vals)
            if spread > worst:
                worst, worst_cell = spread, '%s / %s' % (cell, suffix)
            print('%-34s %-18s %20s %8.1f'
                  % (cell, suffix, ' '.join('%.1f' % v for v in vals), spread), file=out)

    if moved:
        print('\nEXCLUDED -- the unimodal encoder scores moved, so these are two different', file=out)
        print('checkpoints of the same arm, not two runs of one model:', file=out)
        for cell in sorted(moved, key=lambda c: -moved[c][1]):
            metrics, drift = moved[cell]
            detail = '  '.join('%s %s' % (k, '/'.join('%.1f' % v for v in vals))
                               for k, vals in sorted(metrics.items())
                               if k in ENCODER_METRICS and len(vals) > 1)
            print('  %-30s encoder drift %.1f   (%s)' % (cell, drift, detail), file=out)
        print('  Their spread measures training progress between the two evals. Counting it', file=out)
        print('  would inflate the floor and excuse real margins as noise.', file=out)

    if worst_cell is None:
        print('\nNo cell was evaluated twice on one checkpoint -- no floor can be stated, and', file=out)
        print('an absent floor is not a floor of zero. Re-run one finished cell to measure it.', file=out)
        return
    print('\nlargest disagreement between two runs of one checkpoint: %.1f R@1  (%s)'
          % (worst, worst_cell), file=out)
    print('Any margin in the paper below that is inside eval noise and cannot be claimed as a', file=out)
    print('win on this evidence. These replicates are eval-side ONLY -- they share a checkpoint,', file=out)
    print('so they bound eval jitter and say nothing about seed-to-seed training variance,', file=out)
    print('which is larger. Treat this as a LOWER bound on the error bar.', file=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='workdir/e1_zs')
    ap.add_argument('--pivot', action='store_true',
                    help='also print arm x benchmark tables for the aggregator and ITM metrics')
    args = ap.parse_args()
    root = args.root if os.path.isabs(args.root) else os.path.join(ROOT, args.root)
    dirs = sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d))
    if not dirs:
        print('no eval workdirs under %s -- run on the cluster' % args.root, file=sys.stderr)
        return 2
    parsed = 0

    f = lambda v: '--' if v is None else '%.1f' % v
    rows = {}                       # cell -> (cos T-V, cos T-A, aggregator, ITM), for the pivots
    repeats = {}                    # suffix -> [(cell, [every run], spread)], the noise floor
    print('%-34s %8s %8s %8s %8s %8s %8s' %
          ('cell', 'cos T-V', 'cos T-A', 'best 1mod', 'AGGREG', 'GAIN', 'ITM'))
    print('-' * 88)
    for d in dirs:
        got, seen = scan(d)
        if not got:
            continue
        parsed += 1
        name = os.path.basename(d)
        tv, ta = got.get('cosine_TV'), got.get('cosine_TA')
        agg, itm = got.get('ret_area_forward'), got.get('ret_itm_area')
        solo = max([v for v in (tv, ta) if v is not None], default=None)
        # aggregation gain (Delta_agg): the aggregator against the best single modality it
        # fused. Negative gain means the aggregator scores WORSE than one of its own inputs
        # -- fusion destroyed information.
        gain = '%+.1f' % (agg - solo) if (agg is not None and solo is not None) else '--'
        rows[name] = (tv, ta, agg, itm)
        rerun = max((len(v) for v in seen.values()), default=1)
        print('%-34s %8s %8s %8s %8s %8s %8s%s'
              % (name, f(tv), f(ta), f(solo), f(agg), gain, f(itm),
                 '   <- %d runs in the log, reporting the LAST' % rerun if rerun > 1 else ''))
        if rerun > 1:
            # keyed by CELL, not by metric: telling a replicate from a later checkpoint needs
            # this cell's unimodal scores alongside its aggregator and ITM scores
            repeats[name] = {k: v for k, v in seen.items() if len(v) > 1}

    if not parsed:
        # never report an empty comparison as if it were a negative result
        print('\nNO WORKDIR YIELDED METRICS. Expected logs at <workdir>/log/log*.txt --')
        print('check that the eval cells wrote there, and that this is the right --root.')
        return 3
    if args.pivot:
        pivot(rows, 2, 'AGGREGATOR R@1 (the method\'s own score -- centroid vs volume)')
        pivot(rows, 3, 'ITM R@1 (the REPORTED metric, after cross-encoder reranking)')
        pivot(rows, 0, 'cosine T-V R@1 (video alone)')
    noise_floor(repeats)

    print('\nGAIN (the aggregation gain, Delta_agg) is the diagnostic: an aggregator that')
    print('scores below the best modality it was built from -- negative gain -- is destroying')
    print('information. Compare an SCA cell against the released-GRAM cell for the SAME')
    print('benchmark: worse gain on equal or better single-modality features is a fusion')
    print('defect, not a representation defect.')
    print('\nITM is the reported metric. If SCA leads on AGGREG and trails on ITM, the')
    print('reranking stage is where the advantage is lost instead.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
