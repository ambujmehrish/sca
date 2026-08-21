#!/usr/bin/env python3
"""Did each pretrain actually reach the end of its schedule, or stop at a wall clock?

    python3 scripts/audit_training_completion.py
    python3 scripts/audit_training_completion.py --workdir_root workdir_pretrain

Every launcher ran under a wall clock shorter than a full 150k x 5-epoch pretrain and
relied on resubmission to continue. Nothing recorded whether the resubmissions actually
happened, so a run that hit the limit once and was never resubmitted looks exactly like a
finished one: a workdir full of checkpoints, a best_*.pt, results extracted from it.

That matters more than a lost run would. GRAM's released checkpoint is fully trained by its
authors, so any SCA arm that stopped early has been compared against a complete model while
being incomplete itself -- which would depress every number uniformly and look like a method
that only wins marginally.

The check: the highest model_step_*.pt in each workdir against the step count its own
config implies, (clips // batch_size) * epochs. Reports the shortfall, not just a flag,
because a run at 95% of schedule is a different problem from one at 30%.
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# NOT a constant. The subset is called "150k" but the loader skips clips whose video or
# audio failed to download, so the trainable count is lower -- around 134k here. Assuming
# the nominal figure reported every arm as "91% of schedule" when each had in fact run its
# full five epochs. The count is read from the annotation the run itself used; the constant
# survives only as a last resort when that file cannot be opened, and is flagged when used.
NOMINAL_CLIPS = 150000


def find_hps(workdir):
    """The run's recorded config, wherever the trainer put it.

    Not always workdir/hps.json: depending on the launcher it lands in a log/ subdirectory
    or beside the checkpoints. Searching is what makes this usable on the real tree.
    """
    for pat in ('hps.json', 'log/hps.json', '*/hps.json', '*/*/hps.json',
                'config.json', 'log/config.json'):
        hits = sorted(glob.glob(os.path.join(workdir, pat)))
        if hits:
            return hits[0]
    return None


def expected_steps(hps_path, clips):
    """Steps the schedule implies, from the run's own recorded config."""
    if not hps_path:
        return None, None, None
    try:
        hps = json.load(open(hps_path))
    except (ValueError, IOError):
        return None, None, None
    data = hps.get('data_cfg', {})
    train = (data.get('train') or [{}])[0]
    bs = train.get('batch_size')
    ep = train.get('epoch')
    actual = dataset_clips(hps)
    if actual:
        clips = actual          # measured beats assumed
    # an explicit step budget wins over the epoch formula, and some configs set only that
    if train.get('steps'):
        return int(train['steps']), bs, ep
    for key in ('num_train_steps', 'train_steps'):
        if hps.get('run_cfg', {}).get(key):
            return int(hps['run_cfg'][key]), bs, ep
    if not bs or not ep:
        return None, bs, ep
    return (clips // int(bs)) * int(ep), bs, ep


def clips_note(hps_path):
    """Whether the target came from the real annotation or from the nominal fallback."""
    if not hps_path:
        return 'unknown'
    try:
        n = dataset_clips(json.load(open(hps_path)))
    except (ValueError, IOError):
        return 'unknown'
    return ('counted %d clips' % n) if n else 'ASSUMED %d -- annotation unreadable' % NOMINAL_CLIPS


def dataset_clips(hps):
    """Unique clips in the annotation this run actually trained on, or None."""
    train = (hps.get('data_cfg', {}).get('train') or [{}])[0]
    txt = train.get('txt')
    if not txt:
        return None
    txt = os.path.expandvars(txt)
    path = txt if os.path.isabs(txt) else os.path.join(ROOT, txt)
    if not os.path.exists(path):
        return None
    try:
        data = json.load(open(path))
    except (ValueError, IOError):
        return None
    if not isinstance(data, list):
        return None
    ids = set()
    for e in data:
        if isinstance(e, dict):
            for k in ('video_id', 'clip_id', 'id'):
                if k in e:
                    ids.add(str(e[k]))
                    break
    return len(ids) or len(data)


def reached_step(workdir):
    """Highest step among the checkpoints actually on disk."""
    steps = []
    for pat in ('ckpt/model_step_*.pt', 'ckpt/optimizer_step_*.pt'):
        for f in glob.glob(os.path.join(workdir, pat)):
            m = re.search(r'step_(\d+)\.pt$', f)
            if m:
                steps.append(int(m.group(1)))
    return max(steps) if steps else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir_root', default='workdir_pretrain')
    ap.add_argument('--clips', type=int, default=NOMINAL_CLIPS,
                    help='fallback clip count; the annotation is counted when readable')
    ap.add_argument('--tolerance', type=float, default=0.95,
                    help='fraction of the schedule that counts as complete')
    args = ap.parse_args()

    root = args.workdir_root if os.path.isabs(args.workdir_root) \
        else os.path.join(ROOT, args.workdir_root)
    if not os.path.isdir(root):
        print('FATAL: %s not found -- run this on the cluster' % root, file=sys.stderr)
        return 2

    dirs = sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d))
    if not dirs:
        print('no workdirs under %s' % root)
        return 1

    print('%-30s %8s %9s %7s  %s' % ('arm', 'reached', 'expected', 'done', 'note'))
    print('-' * 84)
    short, unknown, complete = [], [], []
    for d in dirs:
        name = os.path.basename(d)
        exp, bs, ep = expected_steps(find_hps(d), args.clips)
        got = reached_step(d)
        if got is None:
            print('%-30s %8s %9s %7s  no checkpoints' % (name, '-', exp or '?', '-'))
            unknown.append(name)
            continue
        if exp is None:
            print('%-30s %8d %9s %7s  no hps.json -- cannot tell the target'
                  % (name, got, '?', '?'))
            unknown.append(name)
            continue
        frac = got / float(exp)
        note = ''
        if frac < args.tolerance:
            note = 'UNDER-TRAINED: %.0f%% of schedule (batch %s x %s epochs)' % (frac * 100, bs, ep)
            short.append((name, got, exp, frac))
        else:
            complete.append(name)
        print('%-30s %8d %9d %6.0f%%  %s' % (name, got, exp, frac * 100, note))

    print('\n' + '=' * 84)
    print('%d complete, %d under-trained, %d undetermined' % (len(complete), len(short), len(unknown)))

    # Uniformity is the real signal even when the target is unknown: a wall clock truncates
    # arms at ragged, differing steps, whereas a schedule ends every arm at the same one.
    counts = {}
    for d in dirs:
        got = reached_step(d)
        if got is not None:
            counts[got] = counts.get(got, 0) + 1
    if counts:
        top, n = max(counts.items(), key=lambda kv: kv[1])
        print('\nstep counts on disk: %s' % ', '.join(
            '%d steps x%d arm(s)' % (k, v) for k, v in sorted(counts.items())))
        if n == len(dirs):
            print('ALL %d arms stopped at the same step (%d) -- consistent with a schedule'
                  % (n, top))
            print('end, not with a wall clock, which would truncate arms at differing steps.')
        else:
            print('%d of %d arms share the modal step %d; the rest differ -- check whether'
                  % (n, len(dirs), top))
            print('the outliers were resubmitted to completion.')
    if short:
        print('\nUnder-trained arms, worst first:')
        for name, got, exp, frac in sorted(short, key=lambda t: t[3]):
            print('  %-30s %d/%d steps (%.0f%%) -- resubmit to continue' % (name, got, exp, frac * 100))
        print('\nAny result extracted from these was measured on an incomplete model, and')
        print('was compared against a GRAM checkpoint its authors trained to completion.')
        return 1
    if unknown:
        print('\n%d arm(s) could not be determined -- inspect them by hand.' % len(unknown))
        return 3
    print('every arm reached its schedule')
    return 0


if __name__ == '__main__':
    sys.exit(main())
