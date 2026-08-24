#!/usr/bin/env python3
"""Table 4: the objective, one component removed at a time -- all at the reported T9 recipe.

    python3 scripts/build_loss_ablation_table.py
    python3 scripts/build_loss_ablation_table.py --out experiments/results/tables_final/table4_loss_ablation.tex

Every arm is T9 with exactly one knob changed, so a row's delta is that component's
contribution and nothing else. L_align is never removed: it IS the retrieval objective (a
model trained without it has nothing to retrieve with), so the ablation covers the removable
terms only:

    t9_qweight_only    full objective (the reported configuration; reference row)
    g11_train_nomask   masked training OFF (mask_p_full=1: no masked view is ever drawn,
                       L_align sees only full-arity centroids; l_mask is identically zero)
    g10_mask0          L_mask agreement term OFF (beta=0; masked views still train L_align)
    g8_sem0            L_sem OFF (alpha=0)
    g6_lambda0         L_unif OFF (lambda=0)

NO L_concept row: the reported configuration has sca_num_concepts=0, so no prototype
memory exists and L_concept never trains -- sca_delta is inert. The g9_concept0 arm proved
it the hard way: its evaluated cells are numerically identical to T9's on every benchmark
(bit-identical run). Presenting delta=0 as an ablation of the reported objective would
fabricate a component that is not in it.

g10 vs g11 is deliberate: together they separate WHERE masking earns its keep -- drawing
masked views for the main loss (g11 removes that) vs the explicit cross-arity agreement
penalty on top (g10 removes only that).

Cells are workdir/e1_frames/<arm>_<bench>, the same directory and the same extraction
(build_paper_table.cell_metrics) as the main tables, so a number cannot differ between
tables. Reported metric: two-stage T->V R@1. Absent cell prints MISSING, never a remembered
number. Delta-bar = mean change vs the full row over the benchmarks both rows have.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_paper_table import ROOT, LABEL, cell_metrics  # noqa: E402

BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')
ROWS = [
    ('full objective (T9)', 't9_qweight_only'),
    ('\\quad w/o masked training views ($p_{\\mathrm{full}}{=}1$)', 'g11_train_nomask'),
    ('\\quad w/o $\\mathcal{L}_{\\mathrm{mask}}$ ($\\beta{=}0$)', 'g10_mask0'),
    ('\\quad w/o $\\mathcal{L}_{\\mathrm{sem}}$ ($\\alpha{=}0$)', 'g8_sem0'),
    ('\\quad w/o $\\mathcal{L}_{\\mathrm{unif}}$ ($\\lambda{=}0$)', 'g6_lambda0'),
]

# harvest-pivot fallback for the aggregation-gain column (positional, like Table 3's):
#   <cell>  cosTV cosTA best1mod AGGREG GAIN ITM [<- note]
_PIVOT = re.compile(r'^(\S+)\s+((?:[-+]?\d+\.\d+\s+){5}[-+]?\d+\.\d+)\s*(?:<-.*)?$')


def _pivot_vals(arm, bench, _cache={}):
    """[cosTV, cosTA, best1mod, AGGREG, GAIN, ITM] from the committed harvest pivot --
    the off-cluster relay of the same eval logs through the same extractor."""
    if 'rows' not in _cache:
        _cache['rows'] = {}
        for txt in sorted(glob.glob(os.path.join(
                ROOT, 'experiments/results/harvest', 'raw_vs_itm_frames*.txt'))):
            for line in open(txt):
                m = _PIVOT.match(line.rstrip())
                if m:
                    _cache['rows'][m.group(1)] = [float(x) for x in m.group(2).split()]
    return _cache['rows'].get('%s_%s' % (arm, bench))


def gain(arm, bench):
    """Aggregation gain (aggregator - best single pathway) for one e1_frames cell."""
    d = os.path.join(ROOT, 'workdir/e1_frames', '%s_%s' % (arm, bench))
    if os.path.isdir(d):
        from raw_vs_itm import scan
        got, _ = scan(d)
        agg = got.get('ret_area_forward')
        solo = max([v for v in (got.get('cosine_TV'), got.get('cosine_TA'))
                    if v is not None], default=None)
        if agg is not None and solo is not None:
            return agg - solo
    vals = _pivot_vals(arm, bench)
    return vals[4] if vals else None


def t2v_r1(arm, bench):
    got = cell_metrics(os.path.join(ROOT, 'workdir/e1_frames', '%s_%s' % (arm, bench)))
    if got:
        return got[0]
    vals = _pivot_vals(arm, bench)
    return vals[5] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='experiments/results/tables_final/table4_loss_ablation.tex')
    args = ap.parse_args()

    vals = {arm: [t2v_r1(arm, b) for b in BENCHES] for _, arm in ROWS}
    gains = {arm: [gain(arm, b) for b in BENCHES] for _, arm in ROWS}
    full = vals[ROWS[0][1]]
    missing = ['%s/%s' % (arm, b) for _, arm in ROWS
               for b, v in zip(BENCHES, vals[arm]) if v is None]

    def dbar(arm):
        pairs = [(v, f) for v, f in zip(vals[arm], full) if v is not None and f is not None]
        if not pairs:
            return None
        return sum(v - f for v, f in pairs) / len(pairs)

    def gbar(arm):
        gs = [g for g in gains[arm] if g is not None]
        return sum(gs) / len(gs) if gs else None

    out = []
    out.append('%% Generated by scripts/build_loss_ablation_table.py -- do not edit numbers.')
    out.append('%% Reference row t9_qweight_only must equal the SCA seed-0 cells behind Tables 1/2.')
    out.append('\\begin{table}[t]')
    out.append('\\centering')
    out.append("\\caption{Ablating the objective at the reported configuration: each row is "
               "T9 with one component removed and nothing else retrained or retuned "
               "(text$\\rightarrow$video R@1, two-stage protocol; $\\bar{\\Delta}$ = mean "
               "change vs.\\ the full objective; $\\bar{\\Delta}_{\\mathrm{agg}}$ = mean "
               "aggregation gain over the five benchmarks, the representation-level effect "
               "the shared reranker compresses). $\\mathcal{L}_{\\mathrm{align}}$ is the "
               "retrieval objective itself and is never removed. Removing the masked "
               "training views and removing $\\mathcal{L}_{\\mathrm{mask}}$ are separate "
               "rows: the former stops drawing reduced-arity views for the alignment loss, "
               "the latter drops only the explicit cross-arity agreement term.}")
    out.append('\\label{tab:loss_ablation}')
    out.append('\\small')
    out.append('\\setlength{\\tabcolsep}{4pt}')
    out.append('\\begin{tabular}{l%s cc}' % ('c' * len(BENCHES)))
    out.append('\\toprule')
    out.append('Objective & %s & $\\bar{\\Delta}$ & $\\bar{\\Delta}_{\\mathrm{agg}}$ \\\\'
               % ' & '.join(LABEL[b] for b in BENCHES))
    out.append('\\midrule')
    for i, (name, arm) in enumerate(ROWS):
        cells = ['MISSING' if v is None else '%.1f' % v for v in vals[arm]]
        if i == 0:
            dc = '--'
        else:
            d = dbar(arm)
            dc = 'MISSING' if d is None else '%+.1f' % d
        g = gbar(arm)
        gc = 'MISSING' if g is None else '%+.1f' % g
        out.append('%s & %s & %s & %s \\\\' % (name, ' & '.join(cells), dc, gc))
        if i == 0:
            out.append('\\midrule')
    out.append('\\bottomrule')
    out.append('\\end{tabular}')
    out.append('\\end{table}')

    text = '\n'.join(out) + '\n'
    path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').write(text)
    print('wrote %s' % args.out)
    if missing:
        print('\n%d cell(s) NOT MEASURED -- MISSING printed:' % len(missing), file=sys.stderr)
        for m in missing:
            print('  ' + m, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
