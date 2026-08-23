#!/usr/bin/env python3
"""Helpers shared by the authors'-code reproductions (HyperGram, PMRL).

Both released repositories are forks of the same VAST/GRAM trunk, so they carry the same
two traps, and a helper that lives in one generator and is copy-pasted into the other is how
one of them silently stops being checked. Kept here once.
"""
import json
import os
import sys

ID_KEYS = ('clip_id', 'video_id', 'image_id', 'image', 'id')


def task_modalities(task):
    """The modality letters of every retrieval group in a task string like ret%tvas%tv%ta."""
    return set(''.join(task.split('%')[1:]))


def anno_ids(txt, limit=None):
    annos = json.load(open(txt))
    if not isinstance(annos, list):
        sys.exit('FATAL: %s is not a list of annotations.' % txt)
    ids = []
    for a in annos:
        for k in ID_KEYS:
            if k in a:
                ids.append(str(a[k]).split('.')[0])
                break
        if limit and len(ids) >= limit:
            break
    if not ids:
        sys.exit('FATAL: no ids found in %s' % txt)
    return ids


def resolve_audio_dir(audio_dir, txt, name, sample=200):
    """An audio directory that satisfies the trunk's hardcoded `<id>.mp3` existence filter.

    data/IndexAnno.py in BOTH released repositories keeps a sample only if
        os.path.exists(os.path.join(d_cfg['audio'], f"{video_id}.mp3"))
    -- the extension is hardcoded. Our audio is .wav, so the dataset builds EMPTY and the
    loader yields zero batches (a torch.cat on an empty list, if you are lucky; an empty
    training epoch if you are not). Their READER is extension-agnostic, so the fix is the
    symlink farm from scripts/hypergram_audio_shim.py.

    Prefers the directory as configured, then the `_mp3link` farm. Refuses otherwise rather
    than handing their code a directory that filters everything away.
    """
    ids = anno_ids(txt, limit=sample)

    def hits(d):
        return sum(1 for i in ids if os.path.exists(os.path.join(d, i + '.mp3')))

    for cand, why in ((audio_dir, 'as configured'),
                      (audio_dir.rstrip('/') + '_mp3link', 'the .mp3 symlink farm')):
        if os.path.isdir(cand) and hits(cand) == len(ids):
            if cand != audio_dir:
                print('  audio [%s] : using %s (%s)' % (name, cand, why))
            return cand
    sys.exit(
        'FATAL: their data/IndexAnno.py keeps a sample only if "<audio_dir>/<id>.mp3" exists --\n'
        '       the extension is hardcoded -- and %s satisfies that for %d of %d sampled ids.\n'
        '       The dataset "%s" would be built EMPTY. Their READER is extension-agnostic, so\n'
        '       build the symlink farm:\n'
        '         python3 scripts/hypergram_audio_shim.py --build \\\n'
        '             --pairs %s:%s\n'
        '       then re-run. Nothing in their code is edited by this.'
        % (audio_dir, hits(audio_dir), len(ids), name, txt, audio_dir))


def expand(obj, data_root, work_root=None):
    """Substitute our ${DATA_ROOT} / ${WORK_ROOT} placeholders throughout a config fragment."""
    s = json.dumps(obj).replace('${DATA_ROOT}', data_root)
    return json.loads(s.replace('${WORK_ROOT}', work_root or os.environ.get('WORK_ROOT', '')))


def absolutise(path, code_dir, what):
    """Their repos ship no datasets/ directory, so relative annotation paths come from ours."""
    if os.path.isabs(path):
        return path
    p = os.path.join(code_dir, path)
    if not os.path.exists(p):
        sys.exit('FATAL: %s %s not found at %s -- their repo ships no datasets/ directory, so '
                 'it has to come from ours.' % (what, path, p))
    return p
