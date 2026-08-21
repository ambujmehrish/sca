#!/usr/bin/env python3
"""How many of our missing VAST clips does an archive actually contain?

    # list one zip without extracting it, then measure the overlap
    unzip -Z1 part009.zip > /tmp/part009.txt
    python3 scripts/check_zip_clip_coverage.py /tmp/part009.txt

    # or point it straight at zips (reads the central directory only, no extraction)
    python3 scripts/check_zip_clip_coverage.py part009.zip part010.zip

Downloading ~88GB to discover the overlap is 3% would be a poor trade, so this measures a
single part first and extrapolates. Clip ids are taken from the file names, which in these
archives follow VAST's `<youtube_id>.<index>.mp4` convention -- the same ids as our
annotation.

Reports three numbers that matter: how many clips the archive holds, how many of those we
already have (wasted bytes), and how many are genuinely missing for us (the yield).
"""
import argparse
import json
import os
import sys
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
VIDEO_EXTS = ('.mp4', '.mkv', '.webm', '.avi', '.mov')


def ids_from_listing(path):
    """Clip ids from a text listing (one path per line) or from a zip's central directory."""
    names = []
    if path.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        except (zipfile.BadZipFile, IOError) as exc:
            print('  ! %s unreadable: %s' % (path, exc), file=sys.stderr)
            return set()
    else:
        names = [l.strip() for l in open(path) if l.strip()]
    out = set()
    for n in names:
        base = os.path.basename(n)
        stem, ext = os.path.splitext(base)
        if ext.lower() in VIDEO_EXTS:
            out.add(stem)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archives', nargs='+', help='zip files, or text listings of their contents')
    ap.add_argument('--annotation',
                    default=os.path.expandvars('$DATA_ROOT/vast27m_150k/annotations150k.json'))
    ap.add_argument('--videos', default=os.path.expandvars('$DATA_ROOT/vast27m_150k/clips'))
    ap.add_argument('--audios', default=os.path.expandvars('$DATA_ROOT/vast27m_150k/audios_wav'))
    ap.add_argument('--total_parts', type=int, default=23,
                    help='parts in the full archive set, for the extrapolation line')
    args = ap.parse_args()

    if not os.path.exists(args.annotation):
        print('FATAL: %s not found -- run on the cluster' % args.annotation, file=sys.stderr)
        return 2
    want = set()
    for e in json.load(open(args.annotation)):
        if isinstance(e, dict):
            for k in ('video_id', 'clip_id', 'id'):
                if k in e:
                    want.add(str(e[k]))
                    break

    def stems(d):
        if not os.path.isdir(d):
            print('FATAL: %s not readable' % d, file=sys.stderr)
            sys.exit(2)
        return {os.path.splitext(f)[0] for f in os.listdir(d)}
    have = stems(args.videos) & stems(args.audios)
    missing = want - have

    archive = set()
    for a in args.archives:
        got = ids_from_listing(a)
        print('%-46s %6d clips' % (os.path.basename(a), len(got)))
        archive |= got

    useful = archive & missing
    already = archive & have
    unknown = archive - want

    print('\nour annotation      : %d clips' % len(want))
    print('we already have     : %d' % len(have))
    print('we are missing      : %d' % len(missing))
    print('\narchive holds       : %d clips' % len(archive))
    print('  already ours      : %d  (redundant)' % len(already))
    print('  not in our subset : %d  (other VAST clips)' % len(unknown))
    print('  MISSING FOR US    : %d  <- the yield' % len(useful))
    if missing:
        print('  covers            : %.1f%% of our gap' % (100.0 * len(useful) / len(missing)))

    if len(args.archives) == 1 and args.total_parts > 1 and useful:
        est = len(useful) * args.total_parts
        print('\nIf the other %d parts are similar: ~%d recoverable, %.0f%% of the gap.'
              % (args.total_parts - 1, min(est, len(missing)),
                 100.0 * min(est, len(missing)) / len(missing)))
        print('Extrapolation only -- parts may overlap each other or be unevenly distributed.')

    if useful:
        out = os.path.join(ROOT, 'datasets/external/recoverable_from_archive.txt')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            fh.write('\n'.join(sorted(useful)) + '\n')
        print('\nwrote %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
