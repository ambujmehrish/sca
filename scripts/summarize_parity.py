#!/usr/bin/env python3
"""Tables for the protocol-parity and reranker-swap runs, read from their eval logs.

    python3 scripts/summarize_parity.py            # print both tables
    python3 scripts/summarize_parity.py --out experiments/results/harvest/parity.txt

Every number is text->video R@1 parsed from a log; a cell with no log prints MISSING.
"""
import argparse
import glob
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_results import parse_log, r1_r10  # noqa: E402

BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')
LABEL = {'msrvtt': 'MSR-VTT', 'didemo': 'DiDeMo', 'activitynet': 'ActivityNet',
         'vatex': 'VATEX', 'audiocaps': 'AudioCaps'}
NAME = {'sca': 'SCALAR', 'gram': 'GRAM', 'hypergram': 'HyperGRAM', 'vast': 'VAST init',
        'barybind': 'BaryBind T-VA'}
HG_REF = 'experiments/results/harvest/hypergram_authors.txt'    # committed 8-frame HyperGRAM row
# metric family suffix -> direction prefix inside its dict
METRICS = {'itm': ('ret_itm_area', 'volume_ITM_T2D'),
           'stage1': ('ret_area_forward', 'volume_T2D'),
           'ext': ('ret_itm_ext', 'volume_ITM_T2D'),
           'ext_stage1': ('ret_ext_candidates', 'volume_T2D')}


def cell(d):
    """{metric: R@1} from the LAST evaluation in a cell directory (ours or BaryBind's logs)."""
    logs = sorted(glob.glob(os.path.join(d, 'log', 'log*.txt'))) + \
        sorted(glob.glob(os.path.join(d, 'run.log')))
    out = {}
    for lg in logs:
        for fam, entries in parse_log(lg).items():
            for key, (suffix, prefix) in METRICS.items():
                if fam.endswith(suffix):
                    r1, _ = r1_r10(entries[-1][1], prefix)
                    if r1 is not None:
                        out[key] = r1
    return out


def recall_at_k(dump_dir):
    """Share of queries whose ground truth is inside the dumped top-k (the reranker's ceiling)."""
    files = glob.glob(os.path.join(dump_dir, 'rerank_*.pt'))
    if len(files) != 1:
        return None
    import torch
    d = torch.load(files[0], map_location='cpu')
    col = {v: j for j, v in enumerate(d['ids'])}
    top = d['dual'].topk(d['itm_rerank_num'], dim=1).indices
    hit = sum(int(col[t] in set(top[i].tolist())) for i, t in enumerate(d['ids_txt']))
    return 100.0 * hit / len(d['ids_txt'])


def fmt(v):
    return 'MISSING' if v is None else '%.1f' % v


def table(title, header, rows, out):
    w = max(len(r[0]) for r in rows) + 2
    cw = max([12] + [len(str(v)) + 2 for _, vals in rows for v in vals] + [len(h) + 2 for h in header])
    out.append(title)
    out.append(' ' * w + ''.join(h.rjust(cw) for h in header))
    for name, vals in rows:
        out.append(name.ljust(w) + ''.join(str(v).rjust(cw) for v in vals))
    out.append('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    par, swp = os.path.join(ROOT, 'workdir/parity'), os.path.join(ROOT, 'workdir/rerank_swap')
    hdr = [LABEL[b] for b in BENCHES]
    lines = []
    hg = os.path.join(ROOT, 'workdir')

    def get(path, key):
        return cell(path).get(key) if os.path.isdir(path) else None

    rows = []
    for fs, fsname in (('fs8', '8 frames'), ('fsbb', 'BaryBind frames')):
        for m in ('sca', 'gram', 'hypergram', 'vast', 'barybind'):
            if m == 'vast' and fs == 'fsbb':
                continue
            vals = []
            for b in BENCHES:
                if m == 'barybind':
                    p = os.path.join(par, 'barybind_%s_%s' % (fs, b))
                elif m == 'hypergram':
                    p = os.path.join(hg, 'hgeval' if fs == 'fs8' else 'hgeval_fsbb', b)
                elif fs == 'fs8':
                    p = os.path.join(swp, 'own_%s_%s' % (m, b))
                else:
                    p = os.path.join(par, '%s_fsbb_%s' % (m, b))
                vals.append(fmt(get(p, 'itm')))
            rows.append(('%s [%s]' % (NAME[m], fsname), vals))
    table('FINAL two-stage T->V R@1 (one environment, final/released checkpoints;\n'
          'SCALAR/GRAM/HyperGRAM/VAST: T-VAS on MSR-VTT and VATEX, T-VA elsewhere; BaryBind: T-VA everywhere;\n'
          'HyperGRAM and BaryBind keep their own recipes\' checkpoint selection on MSR-VTT test)', hdr, rows, lines)

    # HyperGRAM retrained with its authors' code: against the committed 8-frame row
    ref = {}
    if os.path.isfile(os.path.join(ROOT, HG_REF)):
        import re
        bench = None
        for line in open(os.path.join(ROOT, HG_REF)):
            if line.strip().upper() in [b.upper() for b in BENCHES]:
                bench = line.strip().lower()
            m = re.search(r'REPORTED\s+ret_itm\s+T->V R@1\s+([0-9.]+)', line)
            if m and bench:
                ref[bench] = float(m.group(1))
    new = [get(os.path.join(hg, 'hgeval', b), 'itm') for b in BENCHES]
    table('HyperGRAM reproduction at 8 frames: committed -> retrained',
          hdr, [('HyperGRAM', ['%s -> %s' % (fmt(ref.get(b)), fmt(n)) for b, n in zip(BENCHES, new)])], lines)

    # matched T-VA on the two T-VAS benchmarks: the released BaryBind checkpoint is T-VA only
    cells = (('msrvtt', 'fs8', 'MSR-VTT 8f'), ('vatex', 'fs8', 'VATEX 8f'), ('vatex', 'fsbb', 'VATEX 16f'))
    rows = []
    for m in ('sca', 'gram', 'hypergram', 'barybind'):
        vals = []
        for b, fs, _ in cells:
            if m == 'barybind':
                p = os.path.join(par, 'barybind_%s_%s' % (fs, b))
            elif m == 'hypergram':
                p = os.path.join(hg, 'hgeval_tva_%s' % fs, b)
            else:
                p = os.path.join(par, '%s_tva_%s_%s' % (m, fs, b))
            vals.append(fmt(get(p, 'itm')))
        rows.append(('%s T-VA' % NAME[m].replace(' T-VA', ''), vals))
    table('MATCHED T-VA on the T-VAS benchmarks (reported beside the T-VAS rows, never in place of them;\n'
          'BaryBind: released checkpoint, selected by its authors on the MSR-VTT test set)',
          [c[2] for c in cells], rows, lines)

    rows = []
    for m in ('sca', 'gram', 'vast', 'barybind'):
        vals = []
        for b in BENCHES:
            p = (os.path.join(par, 'barybind_fs8_%s' % b) if m == 'barybind'
                 else os.path.join(swp, 'own_%s_%s' % (m, b)))
            vals.append(fmt(get(p, 'stage1')))
        rows.append(('%s stage-1' % NAME[m], vals))
    for m in ('sca', 'gram', 'vast'):
        rows.append(('%s recall@50' % NAME[m],
                     [fmt(recall_at_k(os.path.join(swp, 'dumps', '%s_%s' % (m, b))))
                      for b in BENCHES]))
    table('STAGE 1 at 8 frames: own aggregator R@1 and candidate recall@50', hdr, rows, lines)

    rows = []
    for rr, cand in (('sca', 'sca'), ('gram', 'gram'), ('vast', 'vast'),
                     ('gram', 'sca'), ('sca', 'gram'), ('vast', 'sca')):
        vals = []
        for b in BENCHES:
            if rr == cand:
                vals.append(fmt(get(os.path.join(swp, 'own_%s_%s' % (rr, b)), 'itm')))
            else:
                vals.append(fmt(get(os.path.join(swp, 'swap_%s_reranks_%s_%s' % (rr, cand, b)), 'ext')))
        rows.append(('%s reranks %s top-50' % (NAME[rr], NAME[cand]), vals))
    table('RERANKER SWAP at 8 frames: final T->V R@1', hdr, rows, lines)

    rows = []
    for rr, cand in (('gram', 'sca'), ('sca', 'gram'), ('vast', 'sca')):
        vals = []
        for b in BENCHES:
            own = get(os.path.join(swp, 'own_%s_%s' % (rr, b)), 'itm')
            ctl = get(os.path.join(swp, 'swap_%s_reranks_%s_%s' % (rr, cand, b)), 'itm')
            vals.append('ok' if own is not None and ctl is not None and abs(own - ctl) < 0.05
                        else ('MISSING' if own is None or ctl is None else 'DRIFT %+.1f' % (ctl - own)))
        rows.append(('%s own result reproduced' % NAME[rr], vals))
    table('SELF-CHECK: each swap run recomputes its own-candidate result', hdr, rows, lines)

    text = '\n'.join(lines)
    print(text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text + '\n')
    return 1 if 'MISSING' in text or 'DRIFT' in text else 0


if __name__ == '__main__':
    sys.exit(main())
