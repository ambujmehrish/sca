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
# Three tiers, matching the claims made about each component: the CORE is the objective and
# is not ablatable; the MECHANISM rows carry the robustness pillar and are where effects
# must (and do) show; the REGULARIZERS are labeled as such -- small, free, not load-bearing.
TIERS = [
    ('\\emph{(a) core (not ablatable: $\\mathcal{L}_{\\mathrm{align}}$ is the objective)}', [
        ('full objective (T9)', 't9_qweight_only'),
    ]),
    ('\\emph{(b) mechanism: masked-view training}', [
        ('\\quad w/o masked training views ($p_{\\mathrm{full}}{=}1$)', 'g11_train_nomask'),
        ('\\quad w/o $\\mathcal{L}_{\\mathrm{mask}}$ ($\\beta{=}0$)', 'g10_mask0'),
    ]),
    ('\\emph{(c) regularizers}', [
        ('\\quad w/o $\\mathcal{L}_{\\mathrm{sem}}$ ($\\alpha{=}0$)', 'g8_sem0'),
        ('\\quad w/o $\\mathcal{L}_{\\mathrm{unif}}$ ($\\lambda{=}0$)', 'g6_lambda0'),
    ]),
]
ROWS = [row for _, rows in TIERS for row in rows]

# the missing-modality sweep names T9's cells 'sca'; ablation arms carry their own names
# (SCA_ONLY_ARMS mode of slurm_scripts/missing_eval.sh)
MISSING_PREFIX = {'t9_qweight_only': 'sca'}
# Delta_90 provenance: the reference and mechanism rows get swept (absence = MISSING, a
# hole); the regularizer rows are not swept by design and print '--' (a decision).
SWEEP_EXPECTED = {'t9_qweight_only', 'g11_train_nomask', 'g10_mask0'}

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


def _missing_agg(arm, bench, rate, field='agg', _cache={}):
    """Aggregator R@1 (field='agg') or aggregation gain (field='gain') for one masked cell
    (workdir/e1_missing), pivot fallback."""
    name = '%s_%s_%s' % (MISSING_PREFIX.get(arm, arm), bench, rate)
    d = os.path.join(ROOT, 'workdir/e1_missing', name)
    if os.path.isdir(d):
        from raw_vs_itm import scan
        got, _ = scan(d)
        agg = got.get('ret_area_forward')
        if agg is not None:
            if field == 'agg':
                return agg
            best = max((v for v in (got.get('cosine_TV'), got.get('cosine_TA'))
                        if v is not None), default=None)
            return (agg - best) if best is not None else None
    if 'rows' not in _cache:
        _cache['rows'] = {}
        for txt in sorted(glob.glob(os.path.join(
                ROOT, 'experiments/results/harvest', 'raw_vs_itm_missing*.txt'))):
            for line in open(txt):
                m = _PIVOT.match(line.rstrip())
                if m:
                    _cache['rows'][m.group(1)] = [float(x) for x in m.group(2).split()]
    vals = _cache['rows'].get(name)
    if vals is None:
        return None
    return vals[3] if field == 'agg' else vals[4]   # AGGREG / GAIN pivot columns


def masked_gain(arm):
    """Mean aggregation gain (aggregator minus best unimodal) at r=90% test-time masking,
    over the benchmarks with the cell measured. THIS is the robustness column: the drop
    SLOPE barely separates arms (T9 13.6 / g11 13.8 / g10 12.4 mean drop 0->90), because
    every arm rides the same encoders down -- the masked-training effect lives in the gain
    LEVEL, which stays near zero for T9 (-0.5) and collapses without masked views (-7.6).
    None until the arm has been swept (SCA_ONLY_ARMS mode)."""
    gains = []
    for b in BENCHES:
        v = _missing_agg(arm, b, 'r90', field='gain')
        if v is not None:
            gains.append(v)
    return sum(gains) / len(gains) if gains else None


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
    out.append("\\caption{Ablating the objective: each row is the reported configuration "
               "with one component removed, nothing retrained or retuned. Left block: "
               "two-stage text$\\rightarrow$video R@1 ($\\bar{\\Delta}$ = mean change vs.\\ "
               "the full objective) -- at full modality individual effects sit within seed "
               "variance, because the shared reranker compresses them. The components' "
               "effects are structural: $\\bar{\\Delta}_{\\mathrm{agg}}$ is the mean "
               "aggregation gain (own score vs.\\ best unimodal, Table~\\ref{tab:gain}) "
               "and $\\bar{\\Delta}_{\\mathrm{agg}}^{90}$ the same under 90\\% test-time "
               "masking. Removing the masked training views collapses the gain "
               "($+1.0\\!\\to\\!-5.5$, and $-7.6$ under masking) while the video pathway "
               "is unchanged: masked-view training is what makes the query-conditioned "
               "centroid a positive-gain aggregator. $\\mathcal{L}_{\\mathrm{mask}}$ drops "
               "only the explicit cross-arity agreement term on top of those views; the "
               "regularizers are free and not load-bearing (not swept: '--').}")
    out.append('\\label{tab:loss_ablation}')
    out.append('\\small')
    out.append('\\setlength{\\tabcolsep}{4pt}')
    ncol = 1 + len(BENCHES) + 3
    out.append('\\begin{tabular}{l%s ccc}' % ('c' * len(BENCHES)))
    out.append('\\toprule')
    out.append('Objective & %s & $\\bar{\\Delta}$ & $\\bar{\\Delta}_{\\mathrm{agg}}$ '
               '& $\\bar{\\Delta}_{\\mathrm{agg}}^{90}$ \\\\'
               % ' & '.join(LABEL[b] for b in BENCHES))
    for tier, rows in TIERS:
        out.append('\\midrule')
        out.append('\\multicolumn{%d}{l}{%s} \\\\' % (ncol, tier))
        for name, arm in rows:
            cells = ['MISSING' if v is None else '%.1f' % v for v in vals[arm]]
            if arm == ROWS[0][1]:
                dc = '--'
            else:
                d = dbar(arm)
                dc = 'MISSING' if d is None else '%+.1f' % d
            g = gbar(arm)
            gc = 'MISSING' if g is None else '%+.1f' % g
            md = masked_gain(arm)
            if md is not None:
                mc = '%+.1f' % md
            elif arm in SWEEP_EXPECTED:
                mc = 'MISSING'      # reference + mechanism rows are swept; absence is a hole
            else:
                mc = '--'           # regularizers are not swept BY DECISION (caption)
            out.append('%s & %s & %s & %s & %s \\\\'
                       % (name, ' & '.join(cells), dc, gc, mc))
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
