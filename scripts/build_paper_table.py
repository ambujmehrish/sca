#!/usr/bin/env python3
"""The per-benchmark paper table: both directions, R@1 and R@10, sectioned by geometry.

    python3 scripts/build_paper_table.py --bench msrvtt
    python3 scripts/build_paper_table.py --bench msrvtt --out experiments/results/tables_final/table_msrvtt.tex

Layout follows the paper's existing MSR-VTT table:

    Method | Adapter | Mask | T->V R@1 R@10 | V->T R@1 R@10

    (a) Foundation models            published numbers, reference only
    (b) Gramian-volume alignment     GRAM* (released ckpt), HyperGRAM+ (authors' code)
    (c) Leading-eigenvalue alignment PMRL* (authors' released ckpt)
    (d) Spherical centroid (ours)    SCA, three seeds, mean +- sd on R@1

ROW PROVENANCE, fixed by the project: GRAM and PMRL are their authors' released checkpoints
evaluated here; HyperGRAM is trained from its authors' unmodified code at their published
recipe (they release no checkpoint); SCA is the single reported configuration (T9) over
three seeds. Published numbers appear only in block (a)+(b) reference rows and are never
what SCA is bolded against.

Every measured number is read from an eval log. A cell that cannot be read prints MISSING
and the script exits non-zero -- filling a cell by hand is how a mis-scored number becomes a
claim. The one deliberate exception: HyperGRAM on AudioCaps prints '--', because their
release does not run the audio-anchor benchmark and their paper reports no AudioCaps number
either; that is a scope fact about their method, recorded in the caption, not a missing
measurement of ours.
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_notes import VATEX_NOTE_FULL                          # noqa: E402
from extract_results import parse_log, r1_r10                     # noqa: E402
from parse_authors_eval import canon, parse as parse_authors      # noqa: E402

BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')
LABEL = {'msrvtt': 'MSR-VTT', 'didemo': 'DiDeMo', 'activitynet': 'ActivityNet',
         'vatex': 'VATEX', 'audiocaps': 'AudioCaps'}
# The modality variant each benchmark is evaluated with in OUR protocol; published reference
# rows are read at the same variant so the reference block describes the same task.
TASK_KEY = {'msrvtt': 'T-VAS', 'didemo': 'T-VA', 'activitynet': 'T-VA',
            'vatex': 'T-VAS', 'audiocaps': 'T-VA'}

# (display name, published_rows.json key, adapter)
FOUNDATION = [('ImageBind', 'ImageBind', '--'), ('UMT-L (25M)', 'UMT-L', '--'),
              ('LanguageBind', 'LanguageBind', '--'), ('mPLUG-2', 'mPLUG-2', '--'),
              ('VideoPrism-b', 'VideoPrism-b', '--'), ('VAST (27M)', 'VAST', 'full-FT')]

SCA_SEEDS = ('t9_qweight_only', 's1_t9_seed51', 's2_t9_seed52')

PARAMS_JSON = os.path.join(ROOT, 'experiments/results/tables_final/trainable_params.json')


def params_str(key):
    """Measured trainable-parameter count as '4.8M', or MISSING.

    ONLY OUR OWN RUNS GET A NUMBER. No baseline count is printed, for two measured reasons:

    1. The two counting rules are not interchangeable, so a number from one cannot sit in a
       column next to a number from the other. For a LoRA run the honest count is the set the
       optimizer updated (exp_avg); for a released baseline the only available count is every
       floating tensor in the checkpoint, which also includes buffers and heads that never
       receive gradients. Our own full-FT control is measurable BOTH ways and reads
       1,242,890,226 (optimizer) against 1,397,367,145 (checkpoint) -- an 11% gap on one
       identical model.
    2. The checkpoint rule also counts trunk artifacts that are not part of the method. A
       key-level diff of the three checkpoints shows our HyperGRAM arm and our full-FT arm
       each carry contra_head_d.linear.weight (720,896 params, the depth projection head
       built unconditionally at gram.py:33 and never called in T-VA/T-VAS eval), which the
       released GRAM checkpoint does not. Printing that as HyperGRAM's parameter count would
       attribute our trunk's dead weight to their method.

    So the column states what each row TRAINS ('full-FT' / 'LoRA, 4.8M'), and only the rows
    we trained ourselves carry a figure. GRAM reports 1B for its three encoders alone; we do
    not repeat or contest it. Absent key = MISSING, never a number quoted from a paper.
    """
    if not os.path.exists(PARAMS_JSON):
        return 'MISSING'
    n = json.load(open(PARAMS_JSON)).get(key, {}).get('trainable')
    if n is None:
        return 'MISSING'
    if n >= 1e9:
        return '%.1fB' % (n / 1e9)
    # below 10M one decimal matters (4.8M, not 5M) -- this is the column's whole point
    return '%.1fM' % (n / 1e6) if n < 1e7 else '%.0fM' % (n / 1e6)


def trainable_col(kind, key=None):
    """The merged 'Trainable' cell: what this row trains, with a count only when we measured
    it ourselves. kind is 'none' (published row), 'fullft', or 'lora'."""
    if kind == 'none':
        return '--'
    if kind == 'fullft':
        return 'full-FT'
    return 'LoRA, %s' % params_str(key)

# HyperGRAM's release does not run the audio-anchor benchmark, and their paper reports no
# AudioCaps number. '--' by decision, not MISSING by accident.
ABSENT = {("HyperGRAM$^{\\dagger}$", 'audiocaps')}


def _pub(rows, name, bench, direction):
    """[R@1, R@10] from published_rows.json at our task variant, walking down to simpler
    variants when the paper did not report the full one. None stays None -> dash."""
    block = rows.get(name, {}).get('zeroshot_%s' % direction) or {}
    for variant in (TASK_KEY[bench], 'T-VA', 'T-V'):
        v = block.get('%s|%s' % (LABEL[bench], variant))
        if v and v[0] is not None:
            return v
    return [None, None]


def cell_metrics(workdir):
    """(t2v_r1, t2v_r10, v2t_r1, v2t_r10) from OUR trunk's eval log for one cell."""
    got = None
    for lg in sorted(glob.glob(os.path.join(workdir, 'log', 'log*.txt'))):
        for family, entries in parse_log(lg).items():
            if not family.endswith('ret_itm_area'):
                continue
            for _step, metrics in entries:
                fwd = r1_r10(metrics, 'volume_ITM_T2D')
                bwd = r1_r10(metrics, 'volume_ITM_D2T')
                if fwd[0] is not None:
                    got = (fwd[0], fwd[1], bwd[0], bwd[1])   # LAST run wins, as in raw_vs_itm
    return got


def authors_metrics(run_log):
    """Same tuple from an authors'-code run.log, either fork's dialect."""
    if not os.path.exists(run_log):
        return None
    for (bench, metric), payload in sorted(parse_authors(run_log).items()):
        if not metric.startswith('ret_itm'):
            continue
        c = canon(payload)

        def r10(key):
            rec = c.get(key)
            parts = str(rec).split('/') if rec else []
            return float(parts[2]) if len(parts) == 3 else None
        return (c.get('video_r1'), r10('video_recall'), c.get('txt_r1'), r10('txt_recall'))
    return None


def fmt(v, sd=None):
    if v is None:
        return 'MISSING'
    if sd is None:
        return '%.1f' % v
    return '%.1f\\tiny{$\\pm$%.1f}' % (v, sd)


def gram_cell(b):
    """(t2v_r1, t2v_r10, v2t_r1, v2t_r10) for the released GRAM checkpoint on bench b."""
    for prefix in ('released', 'gram'):
        for root in ('workdir/e1_zs', 'workdir/e1_final'):
            for d in sorted(glob.glob(os.path.join(ROOT, root, '%s*_%s*' % (prefix, b)))) \
                   + sorted(glob.glob(os.path.join(ROOT, root, '%s_%s' % (prefix, b)))):
                if os.path.isdir(d):
                    got = cell_metrics(d)
                    if got:
                        return got, os.path.relpath(d, ROOT)
    # MSR-VTT: the released checkpoint was validated on this pipeline early on and the
    # measurement is RECORDED in the repo -- read from that record, never typed from memory:
    #   | **official ckpt -- THIS pipeline** | ... | **52.5 / 82.5** (D2T 50.5/81.2) |
    if b == 'msrvtt':
        rec = os.path.join(ROOT, 'experiments/results/wave1/validation_official_gram.md')
        if os.path.exists(rec):
            import re
            m = re.search(r'official ckpt[^|]*THIS pipeline[^|]*\|[^|]*\|\s*'
                          r'\*\*([0-9.]+)\s*/\s*([0-9.]+)\*\*\s*'
                          r'\(D2T\s*([0-9.]+)\s*/\s*([0-9.]+)\)', open(rec).read())
            if m:
                vals = tuple(float(m.group(i)) for i in (1, 2, 3, 4))
                return vals, ('recorded validation %s (same checkpoint, same pipeline)'
                              % os.path.relpath(rec, ROOT))
    return None, 'no released-checkpoint cell found for %s' % b


def sca_cell(b):
    """(means, sds, per-seed lists) over the three seeds for bench b; means None if <2."""
    per_seed = [cell_metrics(os.path.join(ROOT, 'workdir/e1_frames', '%s_%s' % (a, b)))
                for a in SCA_SEEDS]
    ok = [p for p in per_seed if p]
    seeds = [[p[i] for p in ok] for i in range(4)]
    if len(ok) < 2:
        return None, None, seeds
    means = tuple(st.mean(v) for v in seeds)
    sds = tuple(st.stdev(seeds[i]) if i in (0, 2) else None for i in range(4))
    return means, sds, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', choices=BENCHES)
    ap.add_argument('--all', action='store_true',
                    help='write every benchmark to experiments/results/tables_final/table_<bench>.tex')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.all:
        rc = 0
        for bench in BENCHES:
            sys.argv = [sys.argv[0], '--bench', bench,
                        '--out', 'experiments/results/tables_final/table_%s.tex' % bench]
            rc = max(rc, main())
        return rc
    if not args.bench:
        ap.error('pass --bench or --all')
    b = args.bench

    pub = json.load(open(os.path.join(ROOT, 'benchmark_eval/published_rows.json')))

    # ---- reference block (published; never bolded, never compared against)
    ref = []
    for disp, key, adapter in FOUNDATION:
        t2v, v2t = _pub(pub, key, b, 't2v'), _pub(pub, key, b, 'v2t')
        ref.append((disp + '$^{\\S}$', adapter, t2v + v2t))

    # ---- measured rows
    missing = []
    gram, gram_src = gram_cell(b)

    pmrl = authors_metrics(os.path.join(ROOT, 'workdir/pmrl_released', b, 'run.log'))
    hg = authors_metrics(os.path.join(ROOT, 'workdir/hgeval', b, 'run.log'))

    sca_cols, sca_sd, seedvals = [], [], []
    per_seed = [cell_metrics(os.path.join(ROOT, 'workdir/e1_frames', '%s_%s' % (a, b)))
                for a in SCA_SEEDS]
    per_seed_ok = [p for p in per_seed if p]
    for i in range(4):
        vals = [p[i] for p in per_seed_ok if p[i] is not None]
        seedvals.append(vals)
        if len(vals) < 2:
            sca_cols.append(None); sca_sd.append(None)
        else:
            sca_cols.append(st.mean(vals))
            sca_sd.append(st.stdev(vals) if i in (0, 2) else None)   # error bar on R@1 only

    rows = [
        ("GRAM$^{\\star}$", trainable_col('fullft'), '\\xmark', gram),
        ("HyperGRAM$^{\\dagger}$", trainable_col('fullft'), '\\xmark', hg),
        ("PMRL$^{\\star}$", trainable_col('fullft'), '\\xmark', pmrl),
    ]

    # ---- render
    out = []
    out.append('%% Generated by scripts/build_paper_table.py --bench %s -- do not edit numbers.' % b)
    out.append('%% GRAM source cell: ' + gram_src)
    out.append('%% SCA seeds (t2v_r1/t2v_r10/v2t_r1/v2t_r10 per seed): ' + json.dumps(seedvals))
    out.append('\\begin{table}[t]')
    out.append('\\centering')
    out.append("\\caption{Zero-shot retrieval on %s. \\emph{Trainable} states what each row "
               "updates: every baseline fully fine-tunes the shared VAST backbone, so we give "
               "a parameter count only for the rows we trained ourselves and measured. "
               "$\\S$: published numbers "
               "(reference only). $\\star$: authors' released checkpoint, evaluated on our "
               "protocol. $\\dagger$: trained from the authors' released code at their "
               "recipe. SCA: three seeds, $\\pm$ sd on R@1. %s}"
               % (LABEL[b], VATEX_NOTE_FULL if b == 'vatex' else ''))
    out.append('\\label{tab:%s}' % b)
    out.append('\\small')
    out.append('\\setlength{\\tabcolsep}{4pt}')
    out.append('\\begin{tabular}{llccccc}')
    out.append('\\toprule')
    out.append(' & & & \\multicolumn{2}{c}{Text $\\rightarrow$ Video} & '
               '\\multicolumn{2}{c}{Video $\\rightarrow$ Text} \\\\')
    out.append('\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}')
    out.append('Method & Trainable & Mask & R@1 & R@10 & R@1 & R@10 \\\\')
    out.append('\\midrule')

    def cells(vals):
        return ' & '.join('--' if v is None else '%.1f' % v for v in vals)

    out.append('\\multicolumn{7}{l}{\\emph{(a) Foundation models}} \\\\')
    for disp, adapter, vals in ref:
        out.append('%s & %s & \\xmark & %s \\\\'
                   % (disp, trainable_col('fullft' if adapter == 'full-FT' else 'none'),
                      cells(vals)))
    out.append('\\midrule')
    out.append('\\multicolumn{7}{l}{\\emph{(b) Gramian-volume alignment}} \\\\')

    # bold = column max over the MEASURED rows only (b-d measured + SCA)
    measured_vals = [r[3] for r in rows if r[3]] + ([tuple(sca_cols)] if all(
        v is not None for v in sca_cols) else [])
    colmax = [max((m[i] for m in measured_vals if m[i] is not None), default=None)
              for i in range(4)]

    def mcells(name, vals, sds=(None,) * 4):
        if vals is None:
            if (name, b) in ABSENT:
                return ' & '.join(['--'] * 4)
            missing.append(name)
            return ' & '.join(['MISSING'] * 4)
        outc = []
        for i, v in enumerate(vals):
            if v is None:
                outc.append('--'); continue
            cell = fmt(v, sds[i])
            if colmax[i] is not None and abs(v - colmax[i]) < 1e-9:
                cell = '\\textbf{%s}' % cell
            outc.append(cell)
        return ' & '.join(outc)

    for name, trainable, mask, vals in rows[:2]:
        out.append('%s & %s & %s & %s \\\\' % (name, trainable, mask, mcells(name, vals)))
    out.append('\\midrule')
    out.append('\\multicolumn{7}{l}{\\emph{(c) Leading-eigenvalue alignment}} \\\\')
    name, trainable, mask, vals = rows[2]
    out.append('%s & %s & %s & %s \\\\' % (name, trainable, mask, mcells(name, vals)))
    out.append('\\midrule')
    out.append('\\multicolumn{7}{l}{\\emph{(d) Spherical centroid alignment (ours)}} \\\\')
    if all(v is None for v in sca_cols):
        missing.append('SCA (no seed cells readable)')
        sca_cells = ' & '.join(['MISSING'] * 4)
    else:
        sca_cells = mcells('SCA', tuple(sca_cols), tuple(sca_sd))
    out.append('\\textbf{SCA} (ours) & %s & \\cmark & %s \\\\'
               % (trainable_col('lora', 'sca_t9'), sca_cells))
    # the full-FT control: T9's exact recipe with use_lora=false (arm f1_t9_fullft), one
    # run. Printed only once its cell exists -- an absent optional row is silence, not a
    # MISSING, because the LoRA row above is the reported configuration either way.
    fullft = cell_metrics(os.path.join(ROOT, 'workdir/e1_frames', 'f1_t9_fullft_%s' % b))
    if fullft:
        out.append('SCA, full-FT (same recipe) & %s & \\cmark & %s \\\\'
                   % (trainable_col('fullft'), ' & '.join('%.1f' % v for v in fullft)))
    out.append('\\bottomrule')
    out.append('\\end{tabular}')
    out.append('\\end{table}')

    # FIELD-COUNT GUARD. A body row with the wrong number of '&' does not fail in LaTeX -- it
    # silently shifts every later cell one column left, which is how a Params column once ate
    # the foundation rows' R@1. Every emitted body row must have exactly NCOL fields.
    ncol = 7
    for line in out:
        if not line.endswith('\\\\') or line.lstrip().startswith(('\\multicolumn', '\\cmidrule',
                                                                  '&')):
            continue
        nf = len(line.rsplit('\\\\', 1)[0].split('&'))
        if nf != ncol:
            sys.exit('FATAL: row has %d fields, expected %d -- a column would shift '
                     'silently:\n  %s' % (nf, ncol, line))

    text = '\n'.join(out) + '\n'
    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w').write(text)
        print('wrote %s' % args.out)
    else:
        print(text)
    if missing:
        print('\n%d row(s) NOT MEASURED -- MISSING is printed, never a remembered number:'
              % len(missing), file=sys.stderr)
        for m in missing:
            print('  ' + m, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
