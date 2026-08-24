#!/usr/bin/env python3
"""Emit the main table from the measured results, never from typed-in numbers.

    python3 scripts/build_main_table.py
    python3 scripts/build_main_table.py --out experiments/results/tables_final/table1_main_all.tex

Reporting policy, fixed by the project: every comparison row is measured by us in one
environment. GRAM is its released checkpoint; HyperGRAM and PMRL are our reproductions; SCA is
the single reported configuration averaged over its three seeds. Published numbers appear only
as a reference block, never as a row SCA is compared against -- the same released GRAM
checkpoint reads 54.8 in its paper and 52.5 here, so a cross-environment difference of one or
two points is not a result, and that is most of the range this field competes in.

Every number is read from a result directory. A cell that has not been evaluated prints as
MISSING rather than being filled from memory, because a table assembled by hand is exactly
where a mis-scored cell becomes a claim -- and scripts/audit_eval_geometry.py has already found
25 of those.
"""
import argparse
import glob
import os
import re
import statistics as st
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_vs_itm import BENCHES, scan  # noqa: E402
from parse_authors_eval import canon, parse as parse_authors  # noqa: E402

# The three seeds of the single reported configuration: same hyperparameters, seed only.
SCA_SEEDS = ('t9_qweight_only', 's1_t9_seed51', 's2_t9_seed52')

# Where each measured row lives: (workdir root, cell prefix). The prefix must identify the
# arm, not just the method -- two checkpoints of one method share a config file.
MEASURED = [
    ('GRAM, released ckpt', 'full-FT', 'workdir/e1_zs', 'released'),
]

# The two rows produced by the AUTHORS' code rather than by our trunk: PMRL is their released
# checkpoint (huggingface.co/xhLiu/PMRL) evaluated on our protocol; HyperGRAM is trained from
# github.com/uta-smile/HyperGram unmodified at their published recipe and evaluated the same
# way. Their numbers are read from the run logs their own eval wrote; when this script runs
# off-cluster, the committed harvest of those logs is the fallback, and either way the source
# is printed into the header. The retired e1_repro reimplementation cells are not consulted.
AUTHORS = [
    ("PMRL (authors' released ckpt)", 'full-FT', 'workdir/pmrl_released', 'pmrl_released'),
    ("HyperGRAM (authors' code)",     'full-FT', 'workdir/hgeval',        'hypergram_authors'),
]

HARVEST = os.path.join(ROOT, 'experiments/results/harvest')


def authors_itm(workdir_root, harvest_name):
    """bench -> reported T->V R@1 for an authors'-code row, plus where it came from.

    Primary: the per-benchmark run.log files their own evaluation wrote (on the cluster).
    Fallback: the committed harvest of those same logs, written by parse_authors_eval via
    scripts/harvest_and_push.sh -- one parser, one format, so the fallback cannot drift from
    the primary. A benchmark absent from both is None and prints MISSING.
    """
    base = workdir_root if os.path.isabs(workdir_root) else os.path.join(ROOT, workdir_root)
    logs = sorted(glob.glob(os.path.join(base, '*', 'run.log')))
    if logs:
        vals = {}
        for lg in logs:
            for (bench, metric), payload in parse_authors(lg).items():
                if metric.startswith('ret_itm'):
                    vals[bench] = canon(payload).get('video_r1')
        return vals, 'run logs under %s' % workdir_root
    txt = os.path.join(HARVEST, harvest_name + '.txt')
    if os.path.exists(txt):
        vals, bench = {}, None
        for line in open(txt):
            head = line.strip().lower()
            if head in LABEL_LOWER:
                bench = head
            m = REPORTED_RE.search(line)
            if m and bench:
                vals[bench] = float(m.group(1))
                bench = None
        return vals, 'committed harvest %s' % os.path.relpath(txt, ROOT)
    return {}, 'NO SOURCE (neither %s nor %s exists)' % (workdir_root, txt)


LABEL_LOWER = {'msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps'}
import re as _re
REPORTED_RE = _re.compile(r'REPORTED\s+ret_itm\s+T->V R@1\s+([0-9.]+)')

# Published numbers, for the reference block only. Never compared against directly.
PUBLISHED = [
    ('ImageBind',      '--',      [36.8, None, None, None, 9.3]),
    ('UMT-L (25M)',    '--',      [40.7, 48.6, 41.9, None, None]),
    ('LanguageBind',   '--',      [44.8, 39.9, 41.0, None, 19.7]),
    ('mPLUG-2',        '--',      [47.1, 45.7, None, None, None]),
    ('VideoPrism-b',   '--',      [51.4, None, 49.6, 62.5, None]),
    ('VAST (27M)',     'full-FT', [50.7, 49.5, 51.4, 82.1, 32.1]),
    ('GRAM',           'full-FT', [54.8, 54.2, 59.0, 83.5, 33.2]),
    ('HyperGRAM',      'full-FT', [56.6, 51.3, 58.2, 79.9, None]),
    ('PMRL',           'full-FT', [54.5, 50.6, 56.0, 80.5, 36.1]),
]

LABEL = {'msrvtt': 'MSR-VTT', 'didemo': 'DiDeMo', 'activitynet': 'ActivityNet',
         'vatex': 'VATEX', 'audiocaps': 'AudioCaps'}


def itm_of(root, cellprefix, bench):
    """The reported metric for one cell, or None. Matches <prefix>*_<bench> so the arm may
    sit between the method name and the benchmark."""
    base = root if os.path.isabs(root) else os.path.join(ROOT, root)
    hits = [d for d in sorted(glob.glob(os.path.join(base, '%s*_%s' % (cellprefix, bench))))
            if os.path.isdir(d)]
    for d in hits:
        got, _ = scan(d)
        if got.get('ret_itm_area') is not None:
            return got['ret_itm_area']
    # Off-cluster fallback: the committed harvest pivots carry one line per cell,
    #   <cell>  cosTV cosTA best1mod AGGREG GAIN ITM [<- annotation]
    # (older committed pivots say TAX in the header; the regex below is positional -- six
    # numeric fields, ITM last -- so both generations parse identically)
    # written by raw_vs_itm from the same eval logs the scan above would read -- the same
    # numbers through the same extractor, only relayed through git. The ITM column is last.
    pat = re.compile(r'^(%s\S*_%s)\s+((?:[-+]?\d+\.\d+\s+){5}[-+]?\d+\.\d+)\s*(?:<-.*)?$'
                     % (re.escape(cellprefix), re.escape(bench)))
    for txt in sorted(glob.glob(os.path.join(ROOT, 'experiments/results/harvest',
                                             'raw_vs_itm*.txt'))):
        for line in open(txt):
            m = pat.match(line.rstrip())
            if m:
                return float(m.group(2).split()[-1])
    return None


def fmt(v, sd=None):
    if v is None:
        return 'MISSING'
    if sd is None:
        return '%.1f' % v
    return '%.1f\\tiny{$\\pm$%.1f}' % (v, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None, help='write LaTeX here; default stdout')
    args = ap.parse_args()

    rows, missing = [], []

    for name, adapter, root, prefix in MEASURED:
        vals = [itm_of(root, prefix, b) for b in BENCHES]
        missing += ['%s/%s' % (name, b) for b, v in zip(BENCHES, vals) if v is None]
        rows.append((name, adapter, [fmt(v) for v in vals], vals))

    sources = []
    for name, adapter, root, harvest_name in AUTHORS:
        got, src = authors_itm(root, harvest_name)
        sources.append('%s: %s' % (name, src))
        vals = [got.get(b) for b in BENCHES]
        missing += ['%s/%s' % (name, b) for b, v in zip(BENCHES, vals) if v is None]
        rows.append((name, adapter, [fmt(v) for v in vals], vals))

    sca_vals, sca_sd, sca_raw = [], [], []
    for b in BENCHES:
        per = [itm_of('workdir/e1_frames', a, b) for a in SCA_SEEDS]
        per = [v for v in per if v is not None]
        sca_raw.append(per)
        if len(per) < 2:
            missing.append('SCA/%s (only %d seed(s))' % (b, len(per)))
            sca_vals.append(None); sca_sd.append(None)
        else:
            sca_vals.append(st.mean(per)); sca_sd.append(st.stdev(per))
    rows.append(('\\textbf{SCA} (ours)', 'LoRA',
                 [fmt(v, s) for v, s in zip(sca_vals, sca_sd)], sca_vals))

    out = []
    out.append('%% Generated by scripts/build_main_table.py -- do not edit numbers by hand.')
    out.append('%%')
    out.append('%% Every comparison row is measured by us in one environment: GRAM is its')
    out.append('%% released checkpoint, PMRL is its authors\' released checkpoint, HyperGRAM')
    out.append('%% is trained from its authors\' unmodified code at their published recipe,')
    out.append('%% and SCA is the single reported configuration over three seeds. Published')
    out.append('%% numbers are a')
    out.append('%% reference block only -- the same released GRAM checkpoint reads 54.8 in its')
    out.append('%% paper and 52.5 here, so a cross-environment gap of a point or two is not a')
    out.append('%% result.')
    out.append('%%')
    for line in sources:
        out.append('%% source -- ' + line)
    out.append('%% SCA seed values: ' + '; '.join(
        '%s %s' % (LABEL[b], '/'.join('%.1f' % v for v in per) if per else 'none')
        for b, per in zip(BENCHES, sca_raw)))
    if missing:
        out.append('%%')
        out.append('%% NOT YET MEASURED -- these print MISSING and must not be filled in by hand:')
        for m in missing:
            out.append('%%   ' + m)
    out.append('\\begin{table*}[t]')
    out.append('\\centering')
    out.append('\\caption{Zero-shot text-to-video retrieval, R@1. Rows in the lower block are '
               'all measured by us in a single environment and on a single protocol: GRAM and '
               'PMRL from their authors\' released checkpoints, HyperGRAM trained from its '
               'authors\' unmodified code at their published recipe, and SCA as one '
               'configuration over three seeds ($\\pm$ standard deviation). The upper block '
               'reproduces published numbers for reference only: the identical released GRAM '
               'checkpoint reads 54.8 in its paper and 52.5 here, and the same environment '
               'shift moves VATEX the other way, so the two blocks are not comparable in '
               'absolute terms.}')
    out.append('\\label{tab:main}')
    out.append('\\small')
    out.append('\\setlength{\\tabcolsep}{5pt}')
    out.append('\\begin{tabular}{ll%s}' % ('c' * len(BENCHES)))
    out.append('\\toprule')
    out.append('Method & Adapter & %s \\\\' % ' & '.join(LABEL[b] for b in BENCHES))
    out.append('\\midrule')
    out.append('\\multicolumn{%d}{l}{\\emph{As published (reference only)}} \\\\' % (2 + len(BENCHES)))
    for name, adapter, vals in PUBLISHED:
        cells = ' & '.join('--' if v is None else '%.1f' % v for v in vals)
        out.append('%s$^{\\S}$ & %s & %s \\\\' % (name, adapter, cells))
    out.append('\\midrule')
    out.append('\\multicolumn{%d}{l}{\\emph{Measured here, one environment}} \\\\' % (2 + len(BENCHES)))
    best = [max((r[3][i] for r in rows if r[3][i] is not None), default=None)
            for i in range(len(BENCHES))]
    for name, adapter, cells, vals in rows:
        marked = ['\\textbf{%s}' % c if (v is not None and best[i] is not None
                                         and abs(v - best[i]) < 1e-9) else c
                  for i, (c, v) in enumerate(zip(cells, vals))]
        out.append('%s$^{\\ast}$ & %s & %s \\\\' % (name, adapter, ' & '.join(marked)))
    out.append('\\bottomrule')
    out.append('\\end{tabular}')
    out.append('\\end{table*}')
    text = '\n'.join(out) + '\n'

    if args.out:
        p = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        open(p, 'w').write(text)
        print('wrote %s' % args.out, file=sys.stderr)
    else:
        print(text)

    if missing:
        print('\n%d cell(s) NOT MEASURED -- the table prints MISSING for them:' % len(missing),
              file=sys.stderr)
        for m in missing:
            print('  ' + m, file=sys.stderr)
        print('Fill them by running the eval, never by typing the number in.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
