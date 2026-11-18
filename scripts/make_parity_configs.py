#!/usr/bin/env python3
"""Eval configs for the protocol-parity runs: our models at BaryBind's frame counts, and
BaryBind's own code on our annotation files and media.

    python3 scripts/make_parity_configs.py ours                      # benchmark_eval/configs_fsbb/
    python3 scripts/make_parity_configs.py barybind --out workdir/barybind_cfg   # needs $DATA_ROOT

Frame sets:
    fs8   8 frames everywhere (our reported protocol)
    fsbb  BaryBind's test frames: MSR-VTT 8, DiDeMo 32, ActivityNet 32, VATEX 16, AudioCaps 8

Only the val frame count (and the batch size, scaled so memory stays flat) differs between
frame sets. The train entry, which only sizes the frame position embedding, stays at 8 frames
as in BaryBind's own DiDeMo config; the embedding is nearest-interpolated at test time.
"""
import argparse
import copy
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')
FRAMESETS = {
    'fs8': {b: 8 for b in BENCHES},
    'fsbb': {'msrvtt': 8, 'didemo': 32, 'activitynet': 32, 'vatex': 16, 'audiocaps': 8},
}
SOURCES = {'sca': 'benchmark_eval/configs_qweight/sca_%s.json',
           'gram': 'benchmark_eval/configs_e1/gram_%s.json'}
BASE_FRAMES = 8


def load(rel):
    return json.load(open(os.path.join(ROOT, rel)))


def with_frames(entry, frames):
    """Copy of a data entry at `frames` test frames, batch scaled to keep frames*batch fixed."""
    e = copy.deepcopy(entry)
    if e.get('vision_sample_num') != BASE_FRAMES:
        raise ValueError('source entry %s has %s frames, expected %d'
                         % (e.get('name'), e.get('vision_sample_num'), BASE_FRAMES))
    e['vision_sample_num'] = frames
    e['batch_size'] = max(1, e['batch_size'] * BASE_FRAMES // frames)
    return e


def ours_config(model, bench, frameset):
    cfg = load(SOURCES[model] % bench)
    frames = FRAMESETS[frameset][bench]
    out = copy.deepcopy(cfg)
    out['data_cfg']['val'] = [with_frames(v, frames) for v in cfg['data_cfg']['val']]
    check_only_frames_changed(cfg, out)
    return out


def check_only_frames_changed(src, new):
    """Everything but val frames/batch must be byte-identical to the source config."""
    a, b = copy.deepcopy(src), copy.deepcopy(new)
    for e in a['data_cfg']['val'] + b['data_cfg']['val']:
        e.pop('vision_sample_num'), e.pop('batch_size')
    if a != b:
        raise AssertionError('frame-set config differs from its source beyond frames/batch')


def absolutize(entry):
    """Expand ${VARS} and make repo-relative annotation paths absolute (their cwd is their repo)."""
    e = copy.deepcopy(entry)
    for k in ('txt', 'vision', 'audio'):
        if k not in e:
            continue
        v = os.path.expandvars(e[k])
        if '$' in v:
            raise KeyError('unset variable in %s=%s' % (k, e[k]))
        if not os.path.isabs(v):
            v = os.path.normpath(os.path.join(os.path.abspath(ROOT), v))
        e[k] = v
    return e


def barybind_config(bench, frameset):
    """BaryBind's defaults + our data entry; only paths, frames and caption length are ours."""
    src = load(SOURCES['sca'] % bench)
    val = [absolutize(with_frames(v, FRAMESETS[frameset][bench])) for v in src['data_cfg']['val']]
    train = []
    for v in val:
        t = copy.deepcopy(v)
        t['vision_sample_num'] = BASE_FRAMES
        t['epoch'] = 1
        train.append(t)
    model_cfg = {'default': './config/barybind/default_model_cfg.json',
                 'itm_rerank_num': src['model_cfg']['itm_rerank_num'],
                 'ret_bidirection_evaluation': True}
    if 'max_caption_len' in src['model_cfg']:
        model_cfg['max_caption_len'] = src['model_cfg']['max_caption_len']
    return {'run_cfg': {'default': './config/barybind/default_run_cfg.json', 'mode': 'testing'},
            'model_cfg': model_cfg,
            'data_cfg': {'train': train, 'val': val}}


def write(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=1)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('which', choices=('ours', 'barybind'))
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    if args.which == 'ours':
        out = args.out or os.path.join(ROOT, 'benchmark_eval/configs_fsbb')
        for model in SOURCES:
            for b in BENCHES:
                p = os.path.join(out, '%s_%s.json' % (model, b))
                write(p, ours_config(model, b, 'fsbb'))
                print('wrote', os.path.relpath(p, ROOT))
        return 0
    if not args.out:
        sys.exit('FATAL: --out is required for barybind configs (they carry absolute paths)')
    for fs in FRAMESETS:
        for b in BENCHES:
            p = os.path.join(args.out, fs, '%s.json' % b)
            write(p, barybind_config(b, fs))
            print('wrote', p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
