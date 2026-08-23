#!/usr/bin/env python3
"""Make our audio directories satisfy HyperGram's hardcoded `.mp3` existence filter.

    python3 scripts/hypergram_audio_shim.py --report            # diagnose only
    python3 scripts/hypergram_audio_shim.py --build             # create the symlink farms

THE PROBLEM. `data/IndexAnno.py:36-80` in their repo builds the sample list by testing

    os.path.exists(os.path.join(d_cfg['audio'], f"{video_id}.mp3"))

for every dataset name it knows -- the extension is hardcoded. Our audio is `.wav`, so the
filter drops every sample and the dataset comes out EMPTY. That is what produced

    val_loader: ret%tva--vatex_ret has 0 batches
    RuntimeError: torch.cat(): expected a non-empty list of Tensors

and the same filter applies to the training block (`finetune_area`), so training would have
been empty too, one guard later.

WHY A SYMLINK FARM IS THE RIGHT FIX. Their READER does not care about the extension:
`data/audio_mapper.py::AudioMapper.read` tries `id_`, `id_+'.wav'`, `.mp3`, `.mkv` in turn and
decodes with `librosa.load`, which reads the container from the file HEADER, not the name. So
a `.wav` file reached through an `.mp3` symlink decodes to exactly the same waveform. Only the
existence check is extension-bound, and satisfying it with symlinks edits none of their code --
the same principle as linking evaluation_tools and pretrained_weights.

This is asserted rather than assumed: --build decodes a sample of the farm through librosa the
way their beats branch does, and refuses to leave a farm behind that does not read.

THE OTHER REASON THIS MATTERS. `AudioMapper.read` returns `torch.zeros(...)` when it cannot
find a file, printing 'not have audios' and continuing. So an id whose audio is missing does
not fail -- it trains on a silent spectrogram. Coverage is therefore reported explicitly here,
because "the job ran" is not evidence that the audio was there.
"""
import argparse
import json
import os
import sys

# The extensions their reader accepts, in the order it tries them.
READ_ORDER = ('', '.wav', '.mp3', '.mkv')
ID_KEYS = ('clip_id', 'video_id', 'image_id', 'image', 'id')


def anno_ids(path):
    annos = json.load(open(path))
    if not isinstance(annos, list):
        sys.exit('FATAL: %s is not a list of annotations.' % path)
    ids = []
    for a in annos:
        if not isinstance(a, dict):
            sys.exit('FATAL: %s contains a non-object entry.' % path)
        for k in ID_KEYS:
            if k in a:
                ids.append(str(a[k]).split('.')[0] if k == 'video_id' else str(a[k]))
                break
        else:
            sys.exit('FATAL: an entry of %s has none of %s' % (path, ', '.join(ID_KEYS)))
    return ids


def find_source(audio_dir, id_):
    """The file their reader would actually open for this id, or None."""
    for ext in READ_ORDER:
        p = os.path.join(audio_dir, id_ + ext)
        if os.path.isfile(p):
            return p
    return None


def survey(audio_dir, ids):
    """How many ids their FILTER accepts, how many their READER could open."""
    filt = sum(1 for i in ids if os.path.exists(os.path.join(audio_dir, i + '.mp3')))
    readable = sum(1 for i in ids if find_source(audio_dir, i))
    exts = {}
    try:
        for name in os.listdir(audio_dir)[:5000]:
            exts[os.path.splitext(name)[1] or '(none)'] = \
                exts.get(os.path.splitext(name)[1] or '(none)', 0) + 1
    except OSError as e:
        sys.exit('FATAL: cannot list %s: %s' % (audio_dir, e))
    return filt, readable, exts


def verify_decodes(paths):
    """Decode through the farm exactly the way their beats branch does."""
    try:
        import librosa
    except ImportError:
        sys.exit('FATAL: librosa is not importable, so the farm cannot be verified. Their\n'
                 '       audio_mapper decodes with librosa.load; a farm that has not been\n'
                 '       decoded once is a guess, and a silent zero-spectrogram is what a bad\n'
                 '       guess turns into.')
    import warnings
    for p in paths:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wav, sr = librosa.load(p, sr=None)
        if wav is None or len(wav) == 0:
            sys.exit('FATAL: %s decoded to an empty waveform through the .mp3 name.' % p)
        print('  decoded %s -> %d samples @ %s Hz' % (os.path.basename(p), len(wav), sr))


def build_farm(audio_dir, ids, out_dir, verify_n):
    os.makedirs(out_dir, exist_ok=True)
    made = existing = missing = 0
    sample = []
    for i in ids:
        dst = os.path.join(out_dir, i + '.mp3')
        if os.path.islink(dst) or os.path.exists(dst):
            existing += 1
            if len(sample) < verify_n:
                sample.append(dst)
            continue
        src = find_source(audio_dir, i)
        if src is None:
            missing += 1
            continue
        os.symlink(os.path.abspath(src), dst)
        made += 1
        if len(sample) < verify_n:
            sample.append(dst)
    return made, existing, missing, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', nargs='+', required=True,
                    help='ANNOTATION_JSON:AUDIO_DIR pairs to survey or build')
    ap.add_argument('--build', action='store_true', help='create <audio_dir>_mp3link farms')
    ap.add_argument('--report', action='store_true',
                    help='survey only, changing nothing (the default; accepted so that the '
                         'documented command works as written)')
    ap.add_argument('--verify_n', type=int, default=3,
                    help='how many farm entries to actually decode (0 disables -- do not)')
    args = ap.parse_args()
    if args.report and args.build:
        sys.exit('FATAL: --report and --build ask for opposite things; pass one.')

    rc = 0
    for pair in args.pairs:
        if ':' not in pair:
            sys.exit('FATAL: %r is not ANNOTATION_JSON:AUDIO_DIR' % pair)
        txt, audio_dir = pair.rsplit(':', 1)
        if not os.path.exists(txt):
            sys.exit('FATAL: %s not found' % txt)
        if not os.path.isdir(audio_dir):
            sys.exit('FATAL: %s is not a directory' % audio_dir)
        ids = anno_ids(txt)
        filt, readable, exts = survey(audio_dir, ids)
        print('\n%s' % txt)
        print('  audio dir      : %s' % audio_dir)
        print('  extensions     : %s' % ', '.join('%s x%d' % (k, v) for k, v in
                                                  sorted(exts.items(), key=lambda kv: -kv[1])[:5]))
        print('  annotation ids : %d' % len(ids))
        print('  THEIR FILTER accepts (<id>.mp3 exists) : %d  (%.1f%%)'
              % (filt, 100.0 * filt / max(1, len(ids))))
        print('  their READER could open                : %d  (%.1f%%)'
              % (readable, 100.0 * readable / max(1, len(ids))))
        if filt == len(ids):
            print('  -> already satisfied, no shim needed')
            continue
        if readable == 0:
            print('  -> NOTHING here is readable by their reader either. A symlink farm cannot '
                  'help; this is the wrong directory or the audio is absent.', file=sys.stderr)
            rc = 1
            continue
        if not args.build:
            print('  -> %d ids would be DROPPED by their filter. Re-run with --build.'
                  % (len(ids) - filt))
            rc = 1
            continue
        out_dir = audio_dir.rstrip('/') + '_mp3link'
        made, existing, missing, sample = build_farm(audio_dir, ids, out_dir, args.verify_n)
        print('  farm           : %s' % out_dir)
        print('  linked %d, already present %d, no source audio %d' % (made, existing, missing))
        if args.verify_n and sample:
            verify_decodes(sample[:args.verify_n])
        cov = (made + existing) / float(max(1, len(ids)))
        print('  coverage       : %.1f%%' % (100 * cov))
        if cov < 1.0:
            print('  WARNING: %d ids have no audio at all. Their reader returns a ZERO '
                  'spectrogram for those rather than failing, so they would train as silence.'
                  % missing, file=sys.stderr)
    return rc


if __name__ == '__main__':
    sys.exit(main())
