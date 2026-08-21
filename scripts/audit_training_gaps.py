#!/usr/bin/env python3
"""Did every arm train the same amount, on the same data, without a hole in the middle?

    python3 scripts/audit_training_gaps.py
    python3 scripts/audit_training_gaps.py --workdir_root workdir_pretrain --verbose

audit_training_completion.py answers one question -- did the run reach the end of its
schedule. That misses three ways a run can be short or incomparable while still ending at
the right step, all of which have already bitten this project once:

  1. A HOLE. A run that died and was resubmitted resumes from its last checkpoint, but if the
     resume used a different config or the launcher restarted it from scratch under the same
     workdir, the checkpoint sequence has a gap or a rewind. The final step still looks right.

  2. CONFIG DRIFT. The config file on disk today is not necessarily the one the run used. An
     arm trained before a config edit is being read as if it had the new setting -- this is
     how the .done-marker no-op went unnoticed for a whole wave.

  3. A CROSS-ARM MISMATCH. Two arms compared in a table must differ in the ONE thing the
     ablation is about. config/sca/pretrain_cfg/sca_paper.json uses batch 128 where every
     ablation uses 256, which is 2x the optimizer steps for the same 5 epochs -- a real
     difference that no completion check would flag.

Everything here is read from what each run RECORDED, never from the config files as they
stand now, and never from an assumed clip count. Where a run recorded nothing, this says so
instead of filling the gap with an assumption.
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def find_hps(workdir):
    for pat in ('hps.json', 'log/hps.json', '*/hps.json', '*/*/hps.json'):
        hits = sorted(glob.glob(os.path.join(workdir, pat)))
        if hits:
            return hits[0]
    return None


def ckpt_steps(workdir):
    """Every saved model step, sorted. The sequence, not just its maximum."""
    steps = set()
    for f in glob.glob(os.path.join(workdir, 'ckpt', 'model_step_*.pt')):
        m = re.search(r'step_(\d+)\.pt$', f)
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)


EVAL_HEADER = re.compile(r'====[- ]evaluation--.+?=====step (\d+)--')


def eval_steps(workdir):
    """Steps at which the trainer ran validation, from its own log.

    This is the load-bearing signal, because the two more obvious ones are unavailable:
    hps.json is written in utils/args.py:206 but num_train_steps is only computed later in
    utils/build_dataloader.py:67, so the recorded target is ALWAYS 0; and with save_best the
    ckpt directory holds a single file, so a checkpoint sequence cannot show continuity.

    Validation fires on (global_step+1) % valid_steps == 0 with
    valid_steps = num_train_steps // valid_freq - 1 (utils/pipeline.py:121), so the spacing
    of these steps recovers the trainer's own target, and their count recovers how much of
    the run happened -- a full run evaluates valid_freq times.
    """
    steps = set()
    for lg in sorted(glob.glob(os.path.join(workdir, 'log', 'log*.txt'))):
        try:
            for line in open(lg, errors='replace'):
                m = EVAL_HEADER.search(line)
                if m:
                    steps.add(int(m.group(1)))
        except IOError:
            continue
    return sorted(steps)


def target_from_evals(steps, valid_freq):
    """(target, spacing) implied by the validation cadence, or (None, None).

    Inverts valid_steps = num_train_steps // valid_freq - 1. Integer division makes this
    exact only up to a remainder of valid_freq, so the result is a target the run must have
    REACHED, never an exact equality to assert.
    """
    if not valid_freq or len(steps) < 2:
        return None, None
    deltas = [b - a for a, b in zip(steps, steps[1:])]
    spacing = max(set(deltas), key=deltas.count)
    if spacing <= 0:
        return None, None
    return (spacing + 1) * int(valid_freq), spacing


def cadence_gaps(steps):
    """Holes in an otherwise regular save cadence.

    The cadence is the MODAL interval, not the mean: one genuine gap would drag a mean far
    enough to hide itself. An interval that is not a whole multiple of the cadence means the
    run was restarted rather than resumed, so it is reported separately from a clean hole.
    """
    if len(steps) < 3:
        return None, []
    deltas = [b - a for a, b in zip(steps, steps[1:])]
    cadence = max(set(deltas), key=deltas.count)
    if cadence <= 0:
        return None, []
    gaps = []
    for a, b in zip(steps, steps[1:]):
        d = b - a
        if d > cadence:
            gaps.append((a, b, d, d % cadence == 0))
    return cadence, gaps


def train_cfg(hps):
    tr = (hps.get('data_cfg', {}).get('train') or [{}])[0]
    rc = hps.get('run_cfg', {})
    return {'batch_size': tr.get('batch_size'), 'epoch': tr.get('epoch'),
            'annotation': os.path.basename(str(tr.get('txt') or '')) or None,
            'lr': rc.get('learning_rate'),
            # the trainer's OWN target, computed from len(dataset) after unloadable media are
            # dropped. The only trustworthy denominator; an annotation count is an upper bound.
            'num_train_steps': rc.get('num_train_steps'),
            'valid_freq': rc.get('valid_freq'),
            'config': rc.get('config') or hps.get('config')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir_root', default='workdir_pretrain')
    ap.add_argument('--verbose', action='store_true', help='list every saved step per arm')
    args = ap.parse_args()

    root = args.workdir_root if os.path.isabs(args.workdir_root) \
        else os.path.join(ROOT, args.workdir_root)
    if not os.path.isdir(root):
        print('FATAL: %s not found -- run this on the cluster' % root, file=sys.stderr)
        return 2
    dirs = sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d))
    if not dirs:
        print('no workdirs under %s' % root, file=sys.stderr)
        return 2

    arms, no_hps, holes, restarts, incomplete, unverified = {}, [], [], [], [], []
    print('%-26s %5s %4s %8s %8s %8s %6s %6s  %s'
          % ('arm', 'batch', 'ep', 'lr', 'target', 'reached', 'evals', 'ckpts', 'annotation'))
    print('-' * 108)
    for d in dirs:
        name = os.path.basename(d)
        hp = find_hps(d)
        steps = ckpt_steps(d)
        evals = eval_steps(d)
        if hp is None:
            print('%-26s %s' % (name, 'NO hps.json -- nothing recorded, cannot verify'))
            no_hps.append(name)
            continue
        try:
            cfg = train_cfg(json.load(open(hp)))
        except (ValueError, IOError) as e:
            print('%-26s hps.json unreadable (%s)' % (name, e))
            no_hps.append(name)
            continue
        arms[name] = cfg
        reached = max(steps + evals) if (steps or evals) else None
        # hps.json always records 0 (written before the value is computed), so the recorded
        # figure is used only when it is a real number, and the validation cadence otherwise
        recorded = cfg['num_train_steps']
        target = int(recorded) if recorded else None
        derived, spacing = target_from_evals(evals, cfg['valid_freq'])
        if target is None:
            target = derived
        cfg['target'] = target
        print('%-26s %5s %4s %8s %8s %8s %6d %6d  %s'
              % (name, cfg['batch_size'], cfg['epoch'], cfg['lr'],
                 ('%d%s' % (target, '' if recorded else '~')) if target else 'UNKNOWN',
                 reached if reached is not None else '-', len(evals), len(steps),
                 cfg['annotation'] or '?'))
        if args.verbose:
            print('%-26s   ckpt steps: %s' % ('', ', '.join(map(str, steps)) or 'none'))
            print('%-26s   eval steps: %s' % ('', ', '.join(map(str, evals)) or 'none'))

        # completeness
        if target is None or reached is None:
            unverified.append((name, 'no step target could be recovered -- hps.json records 0 '
                                     'and the log has too few evaluation blocks'))
        elif reached < target - (spacing or 0):
            incomplete.append((name, reached, target))
        # a full run evaluates valid_freq times; markedly fewer means it stopped early
        if cfg['valid_freq'] and len(evals) and len(evals) < int(cfg['valid_freq']) - 1:
            incomplete.append((name, reached or 0, target or 0))

        # continuity: prefer the evaluation sequence. With save_best the ckpt dir holds one
        # file, so a checkpoint sequence proves nothing and must not be treated as clean.
        seq, seq_kind = (evals, 'evaluation') if len(evals) >= 3 else (steps, 'checkpoint')
        if len(seq) < 3:
            unverified.append((name, 'only %d evaluation and %d checkpoint step(s) -- too few '
                                     'to see a hole in the sequence' % (len(evals), len(steps))))
        else:
            cadence, gaps = cadence_gaps(seq)
            for a, b, delta, clean in gaps:
                (holes if clean else restarts).append((name, a, b, delta, cadence, seq_kind))

    print()
    ok = True

    if incomplete:
        ok = False
        print('SHORT OF THE TRAINING SCHEDULE -- these stopped early:')
        for name, got, want in sorted(set(incomplete)):
            pct = ' (%.0f%%)' % (100.0 * got / want) if want else ''
            print('  %-26s %d/%s steps%s' % (name, got, want or '?', pct))
        print()

    if holes:
        ok = False
        print('HOLES in the step sequence (a resume that skipped, or deleted files):')
        for name, a, b, delta, cadence, kind in holes:
            print('  %-26s %s step %d -> %d (%d, cadence %d)' % (name, kind, a, b, delta, cadence))
        print()

    if restarts:
        ok = False
        print('IRREGULAR intervals -- not a multiple of the cadence, so likely a restart under')
        print('the same workdir rather than a resume. The final step can still look right:')
        for name, a, b, delta, cadence, kind in restarts:
            print('  %-26s %s step %d -> %d (%d, cadence %d)' % (name, kind, a, b, delta, cadence))
        print()

    # cross-arm uniformity: everything an ablation is NOT about must be identical
    if len(arms) > 1:
        for field in ('batch_size', 'epoch', 'annotation', 'num_train_steps'):
            vals = {}
            for name, cfg in arms.items():
                vals.setdefault(cfg[field], []).append(name)
            if len(vals) > 1:
                ok = False
                print('%s DIFFERS ACROSS ARMS -- these are not comparable as they stand:' % field.upper())
                for v, names in sorted(vals.items(), key=lambda kv: -len(kv[1])):
                    head = ', '.join(sorted(names)[:6])
                    more = ' (+%d more)' % (len(names) - 6) if len(names) > 6 else ''
                    print('  %-14s %2d arm(s): %s%s' % (v, len(names), head, more))
                print()

    if no_hps:
        ok = False
        print('%d workdir(s) recorded no config, so nothing about them is verifiable here: %s\n'
              % (len(no_hps), ', '.join(no_hps)))

    # config drift: what the run recorded vs what the file says today
    drift = []
    for name, cfg in arms.items():
        p = cfg['config']
        if not p:
            continue
        path = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not os.path.exists(path):
            drift.append((name, p, 'config file no longer exists'))
            continue
        try:
            now = train_cfg(json.load(open(path)))
        except (ValueError, IOError):
            continue
        for field in ('batch_size', 'epoch', 'lr', 'annotation'):
            if now[field] is not None and cfg[field] != now[field]:
                drift.append((name, p, '%s: ran with %s, file now says %s'
                              % (field, cfg[field], now[field])))
    if drift:
        ok = False
        print('CONFIG DRIFT -- the file on disk is not what the run used; reading the current')
        print('file to describe these runs would misreport them:')
        for name, p, what in drift:
            print('  %-26s %s -- %s' % (name, p, what))
        print()

    # An unverifiable check is not a passing one. The first version of this script treated a
    # missing target and a single checkpoint as "nothing to report" and printed a clean bill
    # of health for 44 arms while both of its main checks had silently skipped.
    if unverified:
        print('COULD NOT BE VERIFIED -- these checks did not run, which is not the same as')
        print('passing. Do not read the summary below as clearing them:')
        for name, why in sorted(set(unverified)):
            print('  %-26s %s' % (name, why))
        print()

    if ok and not unverified:
        print('No gaps found: every arm reached the step target implied by its own validation')
        print('cadence, evaluated the expected number of times, has a continuous step sequence,')
        print('shares batch size / epochs / annotation with every other arm, and ran the config')
        print('its file still holds.')
        return 0
    if ok:
        print('No gaps found among the checks that COULD run, but %d arm(s) above were not'
              % len(set(n for n, _ in unverified)))
        print('verifiable. Treat this as partial, not clean.')
        return 2
    print('Gaps above. Any table row drawn from an affected arm is not comparable until fixed.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
