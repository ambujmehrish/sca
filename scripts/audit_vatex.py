#!/usr/bin/env python3
"""Why is zero-shot VATEX R@1 ~90 here when GRAM publishes 83.5 on a 431-clip test set?

    python3 scripts/audit_vatex.py                                   # annotations + media
    python3 scripts/audit_vatex.py --dump workdir/rerank_swap/dumps/sca_msrvtt   # + gallery-size effect

Checks, each printed with its numbers (CPU only, run on the cluster where the data lives):
  1. test annotation: clips, queries, captions per clip, which caption is the query
  2. subtitles: how much of each clip's subtitle text overlaps its query caption (leakage check),
     against MSR-VTT as the reference benchmark
  3. media: duration of every test video vs the 10 s window in its VATEX id (trimmed or not),
     for both VATEX video folders present on the cluster
  4. overlap of test YouTube ids with the VATEX train file and the T9 run's pretraining annotation (from its hps.json)
  5. (--dump) gallery-size effect: MSR-VTT R@1 on random 431-clip galleries vs the full 1,000
"""
import argparse
import glob
import json
import os
import random
import re
import statistics
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
TEST = 'datasets/annotations/vatex/descs_ret_test_431.json'
TRAIN = 'datasets/annotations/vatex/descs_ret_train_aug.json'
MSRVTT = 'datasets/annotations/msrvtt/descs_ret_test.json'
PRETRAIN_HPS = 'workdir_pretrain/t9_qweight_only/log/hps.json'
YT = re.compile(r'^(?P<yt>[A-Za-z0-9_-]{11})_(?P<s>\d{6})_(?P<e>\d{6})$')
TOK = re.compile(r"[a-z0-9']+")


def load(rel):
    p = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    if not os.path.isfile(p):
        sys.exit('FATAL: %s not found' % p)
    return json.load(open(p))


def captions(entry):
    for k in ('desc', 'caption', 'captions'):
        if k in entry:
            v = entry[k]
            return v if isinstance(v, list) else [v]
    return []


def jaccard(a, b):
    ta, tb = set(TOK.findall((a or '').lower())), set(TOK.findall((b or '').lower()))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def annotation_report(rel, name):
    d = load(rel)
    ncap = [len(captions(e)) for e in d]
    ids = [e.get('video_id') for e in d]
    print('[%s] %s' % (name, rel))
    print('  entries %d   unique clips %d   captions/entry min %d median %s max %d'
          % (len(d), len(set(ids)), min(ncap), statistics.median(ncap), max(ncap)))
    print('  keys in first entry: %s' % sorted(d[0].keys()))
    sub = [e for e in d if e.get('subtitle')]
    print('  entries with a non-empty subtitle: %d / %d' % (len(sub), len(d)))
    if sub:
        j = [jaccard(captions(e)[0], e['subtitle']) for e in sub]
        exact = sum(1 for e in sub if captions(e)[0].strip().lower() in e['subtitle'].lower())
        print('  caption-vs-subtitle token Jaccard: mean %.3f  p90 %.3f  max %.3f   caption '
              'contained verbatim in subtitle: %d' % (statistics.mean(j),
                                                      sorted(j)[int(0.9 * len(j))], max(j), exact))
    return d


def duration(path):
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                            '-of', 'default=nw=1:nk=1', path], capture_output=True, text=True,
                           timeout=60)
        return float(r.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def media_report(test, folder):
    if not os.path.isdir(folder):
        print('[media] %s: not present' % folder)
        return
    files = {os.path.splitext(f)[0]: os.path.join(folder, f) for f in os.listdir(folder)}
    durs, missing, bad = [], 0, 0
    for e in test:
        vid = e['video_id']
        p = files.get(vid)
        if p is None:
            missing += 1
            continue
        d = duration(p)
        if d is None:
            bad += 1
            continue
        m = YT.match(vid)
        want = (int(m.group('e')) - int(m.group('s'))) if m else None
        durs.append((d, want))
    print('[media] %s: %d test clips found, %d missing, %d unreadable' % (folder, len(durs), missing, bad))
    if durs:
        ds = sorted(d for d, _ in durs)
        trimmed = sum(1 for d, w in durs if w is not None and abs(d - w) <= 1.5)
        longer = sum(1 for d, w in durs if w is not None and d > w + 1.5)
        print('  duration s: min %.1f median %.1f p90 %.1f max %.1f' %
              (ds[0], statistics.median(ds), ds[int(0.9 * len(ds))], ds[-1]))
        print('  matches its 10 s id window (+-1.5 s): %d   longer than the window (untrimmed): %d'
              % (trimmed, longer))


def yt_ids(entries):
    out = set()
    for e in entries:
        m = YT.match(str(e.get('video_id', '')))
        out.add(m.group('yt') if m else str(e.get('video_id', ''))[:11])
    return out


def overlap_report(test):
    t = yt_ids(test)
    tr = yt_ids(load(TRAIN))
    print('[overlap] test YouTube ids %d, also in the VATEX train file: %d' % (len(t), len(t & tr)))
    cfg = load(PRETRAIN_HPS)
    for entry in cfg['data_cfg']['train']:
        p = os.path.expandvars(entry.get('txt', ''))
        if not os.path.isfile(p if os.path.isabs(p) else os.path.join(ROOT, p)):
            print('[overlap] pretraining annotation %s not found' % p)
            continue
        pre = load(p)
        ids = {str(e.get('video_id', ''))[:11] for e in pre}
        print('[overlap] pretraining id samples: %s' % [e.get('video_id') for e in pre[:3]])
        print('[overlap] test YouTube ids also in pretraining file %s: %d' % (p, len(t & ids)))


def gallery_report(dump_dir, size, trials):
    import torch
    files = glob.glob(os.path.join(dump_dir, 'rerank_*.pt'))
    if len(files) != 1:
        sys.exit('FATAL: expected one rerank_*.pt in %s' % dump_dir)
    d = torch.load(files[0], map_location='cpu')
    ids, q = list(d['ids']), list(d['ids_txt'])
    col = {v: j for j, v in enumerate(ids)}
    gt = torch.tensor([col[t] for t in q])

    def r1(score, rows, cols):
        s = score[rows][:, cols]
        pos = {c: k for k, c in enumerate(cols)}
        g = torch.tensor([pos[int(gt[r])] for r in rows])
        return 100.0 * (s.argmax(1) == g).float().mean().item()

    full_rows, full_cols = list(range(len(q))), list(range(len(ids)))
    # ITM scores exist only for each query's full-gallery top-k; unscored clips rank below them
    itm = d['itm_fwd'].clone()
    itm[itm == 0] = -1.0
    base = (r1(d['dual'], full_rows, full_cols), r1(itm, full_rows, full_cols))
    rng = random.Random(0)
    sub = []
    for _ in range(trials):
        cols = sorted(rng.sample(full_cols, size))
        keep = set(cols)
        rows = [i for i in full_rows if int(gt[i]) in keep]
        sub.append((r1(d['dual'], rows, cols), r1(itm, rows, cols)))
    print('[gallery] %s: %d clips -> stage-1 R@1 %.1f, final R@1 %.1f' % (files[0], len(ids), *base))
    print('[gallery] random %d-clip galleries (%d draws): stage-1 R@1 %.1f +- %.1f, final R@1 '
          '%.1f +- %.1f (final is a lower bound: clips outside the full top-k stay unscored)'
          % (size, trials, statistics.mean(s for s, _ in sub), statistics.pstdev(s for s, _ in sub),
             statistics.mean(f for _, f in sub), statistics.pstdev(f for _, f in sub)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=os.environ.get('DATA_ROOT', ''))
    ap.add_argument('--dump', default='', help='rerank dump dir of a 1,000-clip MSR-VTT eval')
    ap.add_argument('--size', type=int, default=431)
    ap.add_argument('--trials', type=int, default=200)
    args = ap.parse_args()
    test = annotation_report(TEST, 'VATEX test')
    annotation_report(MSRVTT, 'MSR-VTT test (reference)')
    if args.data_root:
        for sub in ('VATEX/videos_raw', 'VATEX/videos'):
            media_report(test, os.path.join(args.data_root, sub))
    else:
        print('[media] skipped: set DATA_ROOT')
    overlap_report(test)
    if args.dump:
        gallery_report(args.dump, args.size, args.trials)
    return 0


if __name__ == '__main__':
    sys.exit(main())
