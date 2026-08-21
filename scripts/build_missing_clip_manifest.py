#!/usr/bin/env python3
"""Build a re-download manifest for the VAST-27M clips we are missing.

    python3 scripts/build_missing_clip_manifest.py            # needs network + `datasets`
    python3 scripts/build_missing_clip_manifest.py --ids_only # just list what is missing

VAST distributes URLs, not videos: every group downloads the clips from YouTube itself.
GRAM's repo ships only annotations150k.json and points at the VAST repo for the media. Our
13,480 missing clips (F13) are therefore ones whose source was unavailable, or whose
download failed, when this copy was built -- not clips that were never distributed.

it-just-works/vast27m_annotations mirrors the full 27.6M-row VAST table with `id`, `url`,
`begin_s` and `end_s`, and its ids are in our exact format (`G1DRYgjsZTw.63`). This looks
up the missing ids there and writes a manifest that a downloader can consume.

BEFORE RUNNING THE DOWNLOADS, weigh the cost. Recovering clips changes the pretraining
corpus, so every arm would have to be retrained for the comparisons to stay fair -- and the
gap is 9%, worth tenths of an R@1 over five epochs, on a corpus GRAM was subject to the same
attrition on. Documenting 136,674 usable clips in the setup section is the cheaper and more
honest option. This exists so that decision is made against a real recoverable count.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
HF_DATASET = 'it-just-works/vast27m_annotations'


def missing_ids(annotation, videos, audios):
    data = json.load(open(annotation))
    want = set()
    for e in data:
        if isinstance(e, dict):
            for k in ('video_id', 'clip_id', 'id'):
                if k in e:
                    want.add(str(e[k]))
                    break
    def stems(d):
        if not os.path.isdir(d):
            raise SystemExit('FATAL: %s not readable' % d)
        return {os.path.splitext(f)[0] for f in os.listdir(d)}
    have = stems(videos) & stems(audios)
    return want, want - have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotation',
                    default=os.path.expandvars('$DATA_ROOT/vast27m_150k/annotations150k.json'))
    ap.add_argument('--videos', default=os.path.expandvars('$DATA_ROOT/vast27m_150k/clips'))
    ap.add_argument('--audios', default=os.path.expandvars('$DATA_ROOT/vast27m_150k/audios_wav'))
    ap.add_argument('--out', default='datasets/external/missing_clip_manifest.jsonl')
    ap.add_argument('--ids_only', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.annotation):
        print('FATAL: %s not found -- run on the cluster' % args.annotation, file=sys.stderr)
        return 2
    want, missing = missing_ids(args.annotation, args.videos, args.audios)
    print('annotation clips : %d' % len(want))
    print('missing          : %d (%.1f%%)' % (len(missing), 100.0 * len(missing) / len(want)))

    ids_path = os.path.join(ROOT, 'datasets/external/missing_clip_ids.txt')
    os.makedirs(os.path.dirname(ids_path), exist_ok=True)
    with open(ids_path, 'w') as fh:
        fh.write('\n'.join(sorted(missing)) + '\n')
    print('wrote %s' % ids_path)
    if args.ids_only:
        return 0

    try:
        from datasets import load_dataset
    except ImportError:
        print('\nFATAL: the `datasets` package is required to look up URLs.', file=sys.stderr)
        print('       pip install datasets, or rerun with --ids_only.', file=sys.stderr)
        return 2

    print('\nstreaming %s (27.6M rows) for those ids -- this takes a while' % HF_DATASET)
    ds = load_dataset(HF_DATASET, split='train', streaming=True)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    found = 0
    seen = 0
    with open(out_path, 'w') as fh:
        for row in ds:
            seen += 1
            if seen % 2_000_000 == 0:
                print('  scanned %dM rows, matched %d' % (seen // 1_000_000, found))
            if row.get('id') in missing:
                fh.write(json.dumps({'id': row['id'], 'url': row.get('url'),
                                     'begin_s': row.get('begin_s'),
                                     'end_s': row.get('end_s')}) + '\n')
                found += 1
                if found == len(missing):
                    break
    print('\nmatched %d of %d missing ids' % (found, len(missing)))
    print('wrote %s' % out_path)
    if found < len(missing):
        print('%d ids are absent from the mirror as well -- those cannot be recovered here.'
              % (len(missing) - found))
    print('\nA manifest is not a recovery: each URL still has to be alive on YouTube, and')
    print('changing the corpus means retraining every arm for the comparisons to hold.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
