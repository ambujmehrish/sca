#!/usr/bin/env python3
"""Build a VATEX training split matched in size to GRAM's, for a fair finetuned comparison.

    python3 scripts/make_vatex_matched_split.py            # report only, writes nothing
    python3 scripts/make_vatex_matched_split.py --write     # write the matched annotation

The problem. Our `descs_ret_train_aug.json` lists 26,681 unique VATEX clips; GRAM's Tab. 5
reports 14,060 training videos. The annotation file is VAST's, the same one GRAM's config
points at, so the difference is almost certainly download attrition -- they could fetch
fewer videos than we can. Our finetuned VATEX number (94.2 vs their 87.7) would then rest
partly on training data they never had, which is a data advantage, not a method result.

What this does. Counts the clips that are BOTH in the annotation and present on disk --
the set actually trainable, since the loader skips missing videos -- and, if that exceeds
GRAM's 14,060, deterministically subsamples to exactly that many. All captions of a kept
clip are kept, so only the clip roster shrinks.

What it deliberately does not do. It cannot reproduce GRAM's *identity* of clips: their
repo ships no annotation files, so which 14,060 they held is unknowable. A size-matched
split removes the volume advantage, not the sampling difference, and the paper should say
so. Nothing here touches the test split, which is already the exact 431 clips GRAM reports,
so zero-shot VATEX numbers are unaffected either way.
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
GRAM_TRAIN_CLIPS = 14060          # GRAM arXiv:2412.11959v2, Tab. 5, VATEX train column
SRC = 'datasets/annotations/vatex/descs_ret_train_aug.json'
GRAM_ROSTER = 'datasets/annotations/vatex/gram_repo_train_ids.txt'   # from their repo
OUT = 'datasets/annotations/vatex/descs_ret_train_matched14060.json'
VIDEO_EXTS = ('.mp4', '.mkv', '.webm', '.avi', '.mov')


def clip_id(entry):
    for key in ('video_id', 'clip_id', 'id'):
        if key in entry:
            return str(entry[key])
    return None


def on_disk(video_dir):
    """Basenames (without extension) of every video present, or None if the dir is absent."""
    if not video_dir or not os.path.isdir(video_dir):
        return None
    names = set()
    for fname in os.listdir(video_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in VIDEO_EXTS:
            names.add(stem)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--video_dir', default=os.path.expandvars('$DATA_ROOT/VATEX/videos_raw'),
                    help='where the clips live; used to count what is actually trainable')
    ap.add_argument('--target', type=int, default=GRAM_TRAIN_CLIPS)
    ap.add_argument('--write', action='store_true', help='write the split (default: report only)')
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    if not os.path.exists(src):
        print('FATAL: %s not found -- run this on the cluster' % src, file=sys.stderr)
        return 2
    data = json.load(open(src))
    if not isinstance(data, list):
        print('FATAL: expected a list of annotation entries', file=sys.stderr)
        return 2

    by_clip = {}
    for entry in data:
        cid = clip_id(entry)
        if cid is None:
            continue
        by_clip.setdefault(cid, []).append(entry)

    present = on_disk(args.video_dir)
    print('annotation      : %s' % args.src)
    print('  entries       : %d' % len(data))
    print('  unique clips  : %d' % len(by_clip))
    print('video directory : %s' % args.video_dir)
    if present is None:
        print('  NOT READABLE -- cannot tell how many clips are actually trainable.')
        print('  Re-run where the videos live; without this the comparison is undecided.')
        return 3
    trainable = sorted(set(by_clip) & present)
    print('  videos on disk : %d' % len(present))
    print('  trainable clips: %d   (annotation and disk)' % len(trainable))

    # Step 1: restrict to GRAM's own training roster. Their repo publishes
    # datasets/annotations/vatex/descs_ret_train.json (25,991 clips); our _aug file carries
    # 26,681, so it reaches beyond their train split -- most likely into VATEX val. Those
    # extra clips are training data GRAM's protocol never had, and unlike the download
    # attrition below, this part IS exactly correctable.
    roster_path = os.path.join(ROOT, GRAM_ROSTER)
    if os.path.exists(roster_path):
        roster = {line.strip() for line in open(roster_path) if line.strip()}
        outside = sorted(set(trainable) - roster)
        print('GRAM repo roster: %d clips (datasets/annotations/vatex/descs_ret_train.json)'
              % len(roster))
        print('  ours outside it: %d  <- not in their train split at all' % len(outside))
        trainable = sorted(set(trainable) & roster)
        print('  after intersect: %d' % len(trainable))
    else:
        print('GRAM repo roster: %s NOT FOUND -- cannot drop out-of-split clips'
              % GRAM_ROSTER)

    print('GRAM Tab. 5     : %d  (what they could download)' % args.target)

    if len(trainable) <= args.target:
        print('\nNo subsampling needed: we can train on %d clips, at or below GRAM\'s %d.'
              % (len(trainable), args.target))
        print('The 26,681 figure was annotation size, not trainable size -- the finetuned')
        print('VATEX row is comparable as it stands, and F11 should be corrected.')
        return 0

    excess = len(trainable) - args.target
    print('\n%d trainable clips exceeds GRAM\'s %d by %d (%.2fx).'
          % (len(trainable), args.target, excess, len(trainable) / float(args.target)))

    # Deterministic selection: rank by a hash of the clip id. Independent of file order,
    # of the annotation's ordering, and of the Python version's hash seed, so the same
    # split is reproduced anywhere.
    ranked = sorted(trainable, key=lambda c: hashlib.sha256(c.encode()).hexdigest())
    keep = set(ranked[:args.target])
    kept_entries = [e for e in data if clip_id(e) in keep]
    print('selected %d clips -> %d annotation entries' % (len(keep), len(kept_entries)))

    if not args.write:
        print('\n(report only -- pass --write to create the split)')
        return 0

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    outdir = os.path.dirname(out)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)
    json.dump(kept_entries, open(out, 'w'))
    manifest = {
        'source': args.src,
        'source_unique_clips': len(by_clip),
        'videos_on_disk': len(present),
        'trainable_clips': len(trainable),
        'target': args.target,
        'target_source': 'GRAM arXiv:2412.11959v2 Table 5, VATEX train column',
        'selected_clips': len(keep),
        'selected_entries': len(kept_entries),
        'selection': 'deterministic: clips ranked by sha256(clip_id), first N kept',
        'caveat': ('size-matched to GRAM, not identity-matched: their clip roster is not '
                   'published, so which 14060 they held is unknown'),
    }
    # keep the manifest beside the split it describes, whatever --out was given
    mpath = os.path.splitext(out)[0] + '.manifest.json'
    json.dump(manifest, open(mpath, 'w'), indent=1)
    print('wrote %s' % out)
    print('wrote %s' % mpath)
    return 0


if __name__ == '__main__':
    sys.exit(main())
