#!/usr/bin/env python3
"""Audit every annotation file the configs read: sizes, and train/test clip-id overlap.

Run on the cluster, where datasets/annotations/ actually exists:

    python3 scripts/audit_annotations.py

Why this exists. Two defects found on 2026-08-20 came from annotation files rather than
from code: the AudioCaps finetune config trained on its own test annotation, and the
DiDeMo/ActivityNet eval configs truncated paragraph queries. Filename comparison caught the
first, but filenames are weak evidence -- two differently-named files can still share clip
ids, which is leakage that no config inspection would reveal. This checks the ids
themselves, and compares the test-split sizes against the counts GRAM publishes.

GRAM (arXiv:2412.11959v2) Table 5, "Dataset statistics and hyperparameters":

    Benchmark      Train    Val    Test
    AudioCaps          -      -     700
    VGGSound           -      -    5000
    DiDeMo          8394   1065    1003
    ActivityNet    10009      -    4917
    MSR-VTT         9000      -    1000
    VATEX          14060      -     431

A mismatch is not automatically an error -- videos go offline and every group loses a
different subset -- but an unexplained one belongs in the paper's limitations, and a large
one on the test split makes that benchmark's numbers non-comparable.
"""
import glob
import json
import os
import sys
from collections import OrderedDict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# GRAM Table 5, Appendix B.1.
PUBLISHED = OrderedDict([
    ('audiocaps',   {'train': None,  'val': None,  'test': 700}),
    ('vggsound',    {'train': None,  'val': None,  'test': 5000}),
    ('didemo',      {'train': 8394,  'val': 1065,  'test': 1003}),
    ('activitynet', {'train': 10009, 'val': None,  'test': 4917}),
    ('msrvtt',      {'train': 9000,  'val': None,  'test': 1000}),
    ('vatex',       {'train': 14060, 'val': None,  'test': 431}),
])

ID_KEYS = ('video_id', 'clip_id', 'id', 'image_id', 'audio_id')


def load(path):
    """Return (n_entries, set_of_ids) or (None, None) if unreadable."""
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None, None
    try:
        data = json.load(open(full))
    except (ValueError, IOError) as exc:
        print('    ! unreadable: %s' % exc)
        return None, None
    if isinstance(data, dict):
        data = data.get('annotations', data.get('data', list(data.values())))
    if not isinstance(data, list):
        return None, None
    ids = set()
    for entry in data:
        if not isinstance(entry, dict):
            continue
        for key in ID_KEYS:
            if key in entry:
                ids.add(str(entry[key]))
                break
    return len(data), ids


def bench_of(path):
    low = (path or '').lower()
    for name in PUBLISHED:
        if name in low:
            return name
    return None


def collect():
    """benchmark -> role -> set of annotation paths, from every config in the tree."""
    found = {}
    patterns = ('config/*/finetune_cfg/*.json', 'config/*/pretrain_cfg/*.json',
                'benchmark_eval/configs*/*.json')
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            if 'smoke' in path:
                continue
            try:
                cfg = json.load(open(path))
            except (ValueError, IOError):
                continue
            for split in ('train', 'val'):
                for entry in cfg.get('data_cfg', {}).get(split, []):
                    txt = entry.get('txt')
                    bench = bench_of(txt)
                    if not txt or not bench:
                        continue
                    # a split marked training=true is a genuine training set; everything
                    # else is being read for evaluation regardless of which key it sits under
                    role = 'train' if (split == 'train' and entry.get('training')) else 'eval'
                    found.setdefault(bench, {}).setdefault(role, set()).add(txt)
    return found


def main():
    found = collect()
    if not found:
        print('no annotation references found -- run this from the repo root')
        return 1

    problems = []
    missing = []
    checked = 0
    for bench in PUBLISHED:
        if bench not in found:
            continue
        print('\n=== %s ===' % bench)
        cache = {}
        for role in ('train', 'eval'):
            for path in sorted(found[bench].get(role, ())):
                n, ids = load(path)
                cache[path] = (role, n, ids)
                if n is None:
                    print('  %-5s %-58s MISSING (expected on the cluster)' % (role, path))
                    missing.append((bench, role, path))
                    continue
                checked += 1
                uniq = len(ids) if ids else 0
                want = PUBLISHED[bench]['test'] if role == 'eval' else PUBLISHED[bench]['train']
                note = ''
                if want:
                    # compare against unique clips: annotations often carry one row per caption
                    delta = uniq - want
                    note = '  published %s, delta %+d' % (want, delta)
                    if abs(delta) > max(5, 0.02 * want):
                        note += '  <-- CHECK'
                        problems.append((bench, role, path, uniq, want))
                print('  %-5s %-58s entries %-7s unique-ids %-7s%s'
                      % (role, os.path.basename(path), n, uniq, note))

        trains = [(p, v[2]) for p, v in cache.items() if v[0] == 'train' and v[2]]
        evals = [(p, v[2]) for p, v in cache.items() if v[0] == 'eval' and v[2]]
        for tp, tids in trains:
            for ep, eids in evals:
                shared = tids & eids
                if shared:
                    pct = 100.0 * len(shared) / max(1, len(eids))
                    print('  LEAK  %s  and  %s  share %d ids (%.1f%% of the eval split)'
                          % (os.path.basename(tp), os.path.basename(ep), len(shared), pct))
                    problems.append((bench, 'overlap', '%s|%s' % (tp, ep), len(shared), 0))

    print('\n' + '=' * 70)
    print('checked %d annotation file(s); %d could not be read' % (checked, len(missing)))
    if missing:
        # Never report a clean bill of health for files that were never opened: this audit
        # only means something where the data actually is.
        stale = [m for m in missing if 'IscrC_GMEG' in m[2]]
        live = [m for m in missing if 'IscrC_GMEG' not in m[2]]
        if stale:
            print('  %d reference another user\'s tree (legacy imported configs, ignorable):'
                  % len(stale))
            for b, r, p_ in stale[:3]:
                print('    %s %s %s' % (b, r, p_))
            if len(stale) > 3:
                print('    ... and %d more' % (len(stale) - 3))
        if live:
            print('  %d are real paths absent HERE -- rerun on the cluster to check them:'
                  % len(live))
            for b, r, p_ in live:
                print('    %-12s %-5s %s' % (b, r, p_))
    if problems:
        print('%d item(s) need attention:' % len(problems))
        for bench, kind, path, got, want in problems:
            print('  %-12s %-8s %-52s got %s want %s'
                  % (bench, kind, os.path.basename(path), got, want or '-'))
        return 1
    if not checked:
        print('NOTHING WAS VERIFIED -- no annotation file could be read')
        return 2
    if missing:
        print('of the %d file(s) read: sizes within tolerance, no train/eval id overlap.'
              % checked)
        print('this is NOT a clean bill of health -- rerun where the data lives')
        return 3
    print('all annotation sizes within tolerance, no train/eval id overlap')
    return 0


if __name__ == '__main__':
    sys.exit(main())
