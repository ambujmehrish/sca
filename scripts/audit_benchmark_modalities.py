#!/usr/bin/env python3
"""What modalities does each benchmark ACTUALLY carry, measured from disk?

    python3 scripts/audit_benchmark_modalities.py          # the reported eval configs
    python3 scripts/audit_benchmark_modalities.py --configs benchmark_eval/configs_e1/sca_*.json

WHY THIS EXISTS. The setup section wants a sentence of the form "MSR-VTT and VATEX are scored
with video, audio and subtitles; DiDeMo, ActivityNet and AudioCaps with video and audio". It
is tempting to write that from the task strings (ret%tvas / ret%tva), but a task string says
what we ASK FOR, not what the data HAS: an absent subtitle field or a missing .wav does not
raise, it silently degrades that clip's arity. Two independent reviewers have now questioned
the DiDeMo audio row specifically, and the literature genuinely disagrees --

  GRAM (ICLR 2025) Table 5 lists DiDeMo under "Threemodal (T-V-A)", i.e. audio-bearing, and
  we follow GRAM's splits exactly (DiDeMo 1003 / ActivityNet 4917 / MSR-VTT 1000 /
  VATEX 431 / AudioCaps 700 test clips).

  Hendricks et al. (2017) never mention audio for DiDeMo (Flickr/YFCC100M footage), and at
  least one survey marks its audio unavailable.

Citing one against the other settles nothing. This measures our own extracted data, which is
what our numbers were actually produced from, so the sentence in the paper can be a fact
about this paper rather than a claim about a dataset.

Only AudioCaps is filtered to clips with both streams on disk (benchmark_eval/make_configs.py:
75); every other benchmark uses GRAM's annotation and audio directory unfiltered, so coverage
there is an open question until measured.

Reads annotations and stats files. No GPU, no model, no decoding unless soundfile is present
(then a subset is probed for digital silence, which is the failure mode that would make an
audio file exist and still carry nothing).
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SUBTITLE_KEYS = ('subtitle', 'subtitles', 'raw_subtitles')
ID_KEYS = ('video_id', 'clip_id', 'id')
AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.m4a')
SILENCE_PROBE = 200          # clips decoded for the silence check, when soundfile is available


def expand(p):
    p = os.path.expandvars(str(p))
    if '$' in p:             # an unset ${DATA_ROOT} must not read as "nothing on disk"
        return None
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def clip_id(entry):
    for k in ID_KEYS:
        if entry.get(k):
            return str(entry[k])
    return None


def audio_hit(adir, cid):
    """-> path of this clip's audio file, or None. Tries the extensions our pipelines write."""
    for ext in AUDIO_EXTS:
        p = os.path.join(adir, cid + ext)
        if os.path.exists(p):
            return p
    return None


def silence_scan(paths):
    """-> (n_probed, n_silent) using soundfile, or (0, None) if it is unavailable.

    A .wav that ffmpeg wrote from a video with no audio stream is a valid file full of zeros.
    It costs nothing to load and it contributes a zero vector to the centroid, so it would
    inflate any coverage number computed from file existence alone."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return 0, None
    n_silent = 0
    for p in paths:
        try:
            x, _ = sf.read(p, frames=48000, dtype='float32')     # first ~1-3 s is enough
        except (RuntimeError, IOError):
            n_silent += 1                                        # unreadable counts as unusable
            continue
        if x.size == 0 or float(np.abs(x).max()) < 1e-4:
            n_silent += 1
    return len(paths), n_silent


def audit(cfg_path, block):
    txt, adir = expand(block.get('txt')), expand(block.get('audio'))
    if txt is None or not os.path.exists(txt):
        return None, 'annotation not readable: %s' % (block.get('txt'),)
    data = json.load(open(txt))
    if not isinstance(data, list) or not data:
        return None, 'annotation is not a non-empty list'

    skey = next((k for k in SUBTITLE_KEYS
                 if any(isinstance(e, dict) and e.get(k) for e in data[:2000])), None)
    n_sub = sum(1 for e in data if isinstance(e, dict) and skey and e.get(skey)) if skey else 0

    if not block.get('audio'):
        return dict(n=len(data), skey=skey, n_sub=n_sub, adir=None), None
    if adir is None or not os.path.isdir(adir):
        return None, ('config asks for audio at %r but that directory does not exist -- '
                      'source the env rc where the data lives' % block.get('audio'))

    ids = [clip_id(e) for e in data]
    if any(i is None for i in ids):
        return None, ('some entries carry none of %s, so audio files cannot be matched to '
                      'clips' % (ID_KEYS,))
    hits = [audio_hit(adir, i) for i in ids]
    present = [p for p in hits if p]
    n_probed, n_silent = silence_scan(present[:SILENCE_PROBE])
    return dict(n=len(data), skey=skey, n_sub=n_sub, adir=adir, n_audio=len(present),
                n_probed=n_probed, n_silent=n_silent), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='*',
                    help='default: the reported per-benchmark eval configs (configs_qweight)')
    args = ap.parse_args()
    paths = args.configs or sorted(
        glob.glob(os.path.join(ROOT, 'benchmark_eval/configs_qweight/sca_*.json')))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        sys.exit('FATAL: no configs found -- pass --configs explicitly.')

    print('%-14s %-8s %8s %14s %16s  %s'
          % ('benchmark', 'task', 'clips', 'w/ subtitle', 'w/ audio', 'silence probe'))
    print('-' * 96)
    rows, bad = [], []
    for p in paths:
        cfg = json.load(open(p))
        block = (cfg.get('data_cfg', {}).get('val') or [{}])[0]
        name = os.path.basename(p).replace('sca_', '').replace('.json', '')
        got, err = audit(p, block)
        if err:
            print('%-14s %-8s  %s' % (name, block.get('task', '?'), err))
            bad.append(name)
            continue
        sub = ('%d (%s)' % (got['n_sub'], got['skey'])) if got['n_sub'] else '0'
        if got['adir'] is None:
            aud, sil = 'no audio in cfg', '--'
        else:
            aud = '%d (%.1f%%)' % (got['n_audio'], 100.0 * got['n_audio'] / got['n'])
            sil = ('%d/%d silent' % (got['n_silent'], got['n_probed'])
                   if got['n_silent'] is not None else 'soundfile absent')
        print('%-14s %-8s %8d %14s %16s  %s'
              % (name, block.get('task', '?'), got['n'], sub, aud, sil))
        rows.append((name, block.get('task', '?'), got))

    print()
    print('READING THIS. The task string is what we ASK the model to score; the two coverage')
    print('columns are what the data can actually supply. A benchmark whose task says tvas but')
    print('whose subtitle coverage is a fraction of its clips is a MIXED-arity gallery, which')
    print('is the regime the paper is about -- worth stating as a measured number rather than')
    print('describing the benchmark as "providing" a modality.')
    if any(r[2].get('n_silent') for r in rows):
        print()
        print('Some audio files decode to digital silence. Those are files ffmpeg wrote from a')
        print('video with no audio stream: they exist, they load, and they contribute nothing.')
        print('Count them as absent when writing the setup section.')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
