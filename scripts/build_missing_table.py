#!/usr/bin/env python3
"""Table 3: retrieval under test-time missing modalities -- REPRESENTATION-level.

    python3 scripts/build_missing_table.py           # writes main + supplement tables
    python3 scripts/build_missing_table.py --all     # same (kept for the harvest wiring)

Two tables from the same 50 cells (workdir/e1_missing/<model>_<bench>_r<RR>):

  table3_missing.tex        MAIN: each method's OWN aggregation score (SCA: query-weighted
                            centroid; GRAM: Gramian volume) at r = 0..90% masking. The
                            aggregator is the component the compared papers contribute, and
                            the only stage whose treatment of modality presence differs
                            across methods -- the stage the masking manipulation tests.
  table3_supp_twostage.tex  SUPPLEMENT: the two-stage (reranked) metric on the identical
                            cells, with each method's own video-only cosine at r=90%
                            alongside. It documents WHY the main table is stage-1: the
                            cross-encoder reranker collapses every method onto its video
                            pathway under masking (ITM ~ cos T-V at r=90 for all), erasing
                            aggregator differences.

WORDING OF THE RERANKER CLAIM -- do not restore the old phrasing. These captions used to say
"one frozen cross-encoder shared by all methods". Both halves are false and a reviewer can
check either one:

  NOT FROZEN.  itm_head is not in backbone_prefixes, so it receives gradients whenever
               itm_ratio > 0 (0.1 in our config, matching GRAM Eq. 8 and HyperGRAM Eq. 11).
               A checkpoint diff shows itm_head deltas of 0.002-0.009 after training.
  NOT SHARED.  Each method reranks with the head in ITS OWN checkpoint. Only the VAST
               foundation checkpoint they all start from is common.

What IS true, and is what the argument actually needs:
  - Neither method's reranker is ever trained on an incomplete modality set. Ours conditions
    on condition_feats_{va,vas} built from the UNMASKED encoder outputs (sca.py:_itm_loss);
    the masking lives on the centroid via present_M and never reaches that branch. GRAM's
    trivially never sees one either, since its pipeline discards incomplete clips.
  - MEASURED: as r grows, each method's two-stage R@1 converges to its own video-only cosine
    (the cos_TV@90% column) -- which is the fact the supplement table shows.

Never phrase this as removing the reported metric: the two-stage numbers are all published
in the supplement; the main table reports the stage the manipulation actually measures.

Rows: SCA and GRAM (released ckpt) only. PMRL is excluded BY MEASUREMENT: its released
scoring does not reproduce through our masking harness at r=0 (49.8 vs 54.3 R@1 on MSR-VTT
through its own repo), so masked cells would compare an unanchored implementation.
HyperGRAM's release has no missing-modality path at all. Both exclusions are in the caption.

The r=0 column doubles as the control: byte-identical to the standard eval path, it must
agree with the main tables. Cells are read from the eval logs on the cluster; off-cluster
the committed harvest pivot (raw_vs_itm_missing.txt) carries the same numbers through the
same extractor, parsed positionally -- MISSING is printed for anything unmeasured, never a
remembered number.
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_vs_itm import scan  # noqa: E402

BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')
LABEL = {'msrvtt': 'MSR-VTT', 'didemo': 'DiDeMo', 'activitynet': 'ActivityNet',
         'vatex': 'VATEX', 'audiocaps': 'AudioCaps'}
RATES = ('r00', 'r25', 'r50', 'r75', 'r90')
RATE_LABEL = {'r00': '0\\%', 'r25': '25\\%', 'r50': '50\\%', 'r75': '75\\%', 'r90': '90\\%'}
ROWS = [('\\textbf{SCA} (ours)', 'sca'), ('GRAM$^{\\star}$', 'gram')]

# harvest-pivot fallback: <cell>  cosTV cosTA best1mod AGGREG GAIN ITM [<- note]
# (older committed pivots title the GAIN column TAX; the parse is positional either way)
_PIVOT = re.compile(r'^(\S+)\s+((?:[-+]?\d+\.\d+\s+){5}[-+]?\d+\.\d+)\s*(?:<-.*)?$')
_FIELD = {'agg': 3, 'itm': 5, 'cos_tv': 0}


def _pivot_rows():
    rows = {}
    for txt in sorted(glob.glob(os.path.join(ROOT, 'experiments/results/harvest',
                                             'raw_vs_itm_missing*.txt'))):
        for line in open(txt):
            m = _PIVOT.match(line.rstrip())
            if m:
                rows[m.group(1)] = [float(x) for x in m.group(2).split()]
    return rows


def cell(model, bench, rate, _cache={}):
    """{'agg', 'itm', 'cos_tv'} for one cell, from eval logs or the committed pivot."""
    d = os.path.join(ROOT, 'workdir/e1_missing', '%s_%s_%s' % (model, bench, rate))
    if os.path.isdir(d):
        got, _ = scan(d)
        if got.get('ret_area_forward') is not None:
            return {'agg': got.get('ret_area_forward'), 'itm': got.get('ret_itm_area'),
                    'cos_tv': got.get('cosine_TV')}
    if 'rows' not in _cache:
        _cache['rows'] = _pivot_rows()
    vals = _cache['rows'].get('%s_%s_%s' % (model, bench, rate))
    if vals is None:
        return None
    return {k: vals[i] for k, i in _FIELD.items()}


MAIN_CAPTION = (
    "\\caption{\\textbf{Who retrieves better as modalities vanish:} absolute "
    "text$\\rightarrow$video R@1 of each method's \\emph{own} aggregation score as a "
    "fraction $r$ of gallery clips loses one modality (deterministic per-clip masks, "
    "identical for both methods; $r{=}0$ reproduces the main protocol). Whether each "
    "method's fusion stays \\emph{profitable} against its own unimodal score is the "
    "complementary question, answered in Fig.~\\ref{fig:gain_vs_mask}. $\\star$: authors' "
    "released checkpoint, its volume reduced exactly to each clip's present modalities -- "
    "the released implementation cannot represent missing modalities and discards such "
    "clips. Under masking the cross-encoder reranking stage collapses every method onto its "
    "own video-only pathway, which is analyzed in the supplement; PMRL is excluded because its "
    "released scoring does not reproduce at $r{=}0$ through the masking harness, and "
    "HyperGRAM's release has no missing-modality path.}")

SUPP_CAPTION = (
    "\\caption{\\textbf{Why the main table reports the representation stage:} two-stage "
    "(reranked) R@1 on the identical masked cells of Table~\\ref{tab:missing}, with each "
    "method's own video-only cosine at $r{=}90\\%$. Each method reranks with the "
    "cross-encoder in its own checkpoint, all descending from the same VAST foundation "
    "model and none of them ever trained on an incomplete modality set. As $r$ grows every "
    "method's two-stage score converges onto its own video-only cosine (last column), "
    "erasing the aggregator differences of Table~\\ref{tab:missing}: end-to-end robustness "
    "in a two-stage pipeline is capped by the reranking stage, so it must originate in the "
    "representation stage.}")


def build(metric, out_rel, caption, label, with_cos90):
    vals, missing = {}, []
    for b in BENCHES:
        for _, m in ROWS:
            got = [cell(m, b, r) for r in RATES]
            vals[(m, b)] = got
            missing += ['%s/%s/%s' % (m, b, r) for r, g in zip(RATES, got)
                        if g is None or g.get(metric) is None]

    ncol = 2 + len(RATES) + (1 if with_cos90 else 0)
    out = ['%% Generated by scripts/build_missing_table.py -- do not edit numbers.',
           '%% CONTROL: the 0%% column must agree with the main tables per model.',
           '\\begin{table}[t]', '\\centering', caption, '\\label{%s}' % label,
           '\\small', '\\setlength{\\tabcolsep}{4.5pt}',
           '\\begin{tabular}{ll%s%s}' % ('c' * len(RATES), 'c' if with_cos90 else ''),
           '\\toprule',
           ' & Method & %s%s \\\\' % (' & '.join(RATE_LABEL[r] for r in RATES),
                                      ' & cos$_{TV}$@90\\%' if with_cos90 else '')]
    for b in BENCHES:
        out.append('\\midrule')
        got = {m: vals[(m, b)] for _, m in ROWS}
        for i, (name, m) in enumerate(ROWS):
            cells = []
            for j, r in enumerate(RATES):
                g = got[m][j]
                v = g.get(metric) if g else None
                if v is None:
                    cells.append('MISSING')
                    continue
                others = [got[m2][j].get(metric) for _, m2 in ROWS
                          if got[m2][j] and got[m2][j].get(metric) is not None]
                c = '%.1f' % v
                if others and abs(v - max(others)) < 1e-9:
                    c = '\\textbf{%s}' % c
                cells.append(c)
            if with_cos90:
                g90 = got[m][-1]
                cells.append('%.1f' % g90['cos_tv']
                             if g90 and g90.get('cos_tv') is not None else 'MISSING')
            out.append('%s & %s & %s \\\\' % (LABEL[b] if i == 0 else '', name,
                                              ' & '.join(cells)))
        # the margin row: SCA minus GRAM at each rate, the robustness statement itself
        margins = []
        for j in range(len(RATES)):
            a, g = got['sca'][j], got['gram'][j]
            if a and g and a.get(metric) is not None and g.get(metric) is not None:
                margins.append('%+.1f' % (a[metric] - g[metric]))
            else:
                margins.append('--')
        out.append('\\rowcolor{gray!10} & \\emph{margin} & %s%s \\\\'
                   % (' & '.join(margins), ' & ' if with_cos90 else ''))
    out += ['\\bottomrule', '\\end{tabular}', '\\end{table}']

    path = os.path.join(ROOT, out_rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').write('\n'.join(out) + '\n')
    print('wrote %s' % out_rel)
    if missing:
        print('%s: %d cell(s) NOT MEASURED (MISSING printed): %s'
              % (out_rel, len(missing), ', '.join(missing[:8])), file=sys.stderr)
    return 1 if missing else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='kept for harvest wiring (default)')
    ap.parse_args()
    rc = build('agg', 'experiments/results/tables_final/table3_missing.tex',
               MAIN_CAPTION, 'tab:missing', with_cos90=False)
    rc = max(rc, build('itm', 'experiments/results/tables_final/table3_supp_twostage.tex',
                       SUPP_CAPTION, 'tab:missing_twostage', with_cos90=True))
    return rc


if __name__ == '__main__':
    sys.exit(main())
