#!/usr/bin/env python3
"""Which of our missing VAST-27M clips could an external mirror actually restore?

    python3 scripts/check_missing_clip_recovery.py

Our annotation lists 150,154 clips and 136,674 have both video and audio on disk, so 13,480
are unavailable (F13). This intersects that missing set with a mirror's file listing to say
how many are genuinely recoverable, rather than guessing from the mirror's name.

Ships with the listing for Tensorlong/VAST-150k-clips, which despite its name holds ~1.4k
clips of which 88 are sub-1KB dead downloads. Point --mirror at another listing (one clip
id per line) to check a different source.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DEFAULT_MIRROR = 'datasets/external/hf_tensorlong_vast150k_clips.txt'


def ids_from_annotation(path):
    data = json.load(open(path))
    out = set()
    for e in data:
        if isinstance(e, dict):
            for k in ('video_id', 'clip_id', 'id'):
                if k in e:
                    out.add(str(e[k]))
                    break
    return out


def stems(directory):
    if not os.path.isdir(directory):
        return None
    return {os.path.splitext(f)[0] for f in os.listdir(directory)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotation',
                    default=os.path.expandvars('$DATA_ROOT/vast27m_150k/annotations150k.json'))
    ap.add_argument('--videos', default=os.path.expandvars('$DATA_ROOT/vast27m_150k/clips'))
    ap.add_argument('--audios', default=os.path.expandvars('$DATA_ROOT/vast27m_150k/audios_wav'))
    ap.add_argument('--mirror', default=DEFAULT_MIRROR)
    args = ap.parse_args()

    if not os.path.exists(args.annotation):
        print('FATAL: %s not found -- run on the cluster' % args.annotation, file=sys.stderr)
        return 2
    want = ids_from_annotation(args.annotation)
    have_v, have_a = stems(args.videos), stems(args.audios)
    if have_v is None or have_a is None:
        print('FATAL: clips or audios directory unreadable', file=sys.stderr)
        return 2
    trainable = want & have_v & have_a
    missing = want - trainable

    mirror_path = args.mirror if os.path.isabs(args.mirror) else os.path.join(ROOT, args.mirror)
    mirror = set()
    if os.path.exists(mirror_path):
        mirror = {l.strip() for l in open(mirror_path)
                  if l.strip() and not l.startswith('#')}
    else:
        print('mirror listing %s not found' % args.mirror, file=sys.stderr)

    recoverable = missing & mirror
    print('annotation clips      : %d' % len(want))
    print('trainable now         : %d' % len(trainable))
    print('missing               : %d' % len(missing))
    print('mirror holds          : %d' % len(mirror))
    print('  of which we lack    : %d' % len(recoverable))
    if missing:
        print('  covers              : %.1f%% of the gap' % (100.0 * len(recoverable) / len(missing)))
    if recoverable:
        out = os.path.join(ROOT, 'datasets/external/recoverable_clip_ids.txt')
        with open(out, 'w') as fh:
            fh.write('\n'.join(sorted(recoverable)) + '\n')
        print('\nwrote %s' % out)
        print('Whether that is worth fetching depends on the fraction above; a few percent')
        print('will not move a 5-epoch pretrain, and the gap is probably symmetric with')
        print("GRAM's own attrition (RECIPE_AUDIT F13).")
    else:
        print('\nNothing recoverable from this mirror.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
