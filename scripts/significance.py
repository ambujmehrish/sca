#!/usr/bin/env python3
"""Statistical support for the paper's claims -- computed from harvested cells, no new runs.

    python3 scripts/significance.py

Three honest analyses, each matched to the data that actually exists:

1. SEED STATISTICS (SCA, n=3 seeds). Mean, sd, and the 95% t-interval (t_{2,0.975}=4.303)
   per benchmark for the reported two-stage T->V R@1. A baseline's single released-ckpt
   number lying outside SCA's interval is evidence the ordering is not seed luck. The
   baselines have n=1 BY NATURE (a released checkpoint has no seeds), so no two-sample test
   is possible -- stated rather than faked.

2. EXACT SIGN TESTS on cell-level wins, at two granularities:
   - Table 3 (masked aggregator): SCA vs GRAM over 25 rate x benchmark cells, and
     conservatively over 5 benchmarks (treating each benchmark as one unit, since nested
     masks correlate cells within a benchmark).
   - Gain table: the count of negative baseline gains over 14 method x benchmark cells,
     against the null that fusion helps or hurts with equal probability.
   Exact binomial, one-sided; the dependence caveat is printed with the number.

3. NOISE FLOOR: the measured eval-repeat spread (same checkpoint, same config, repeated
   evals) from the harvest -- the resolution limit below which margins are not results.

Output: experiments/results/tables_final/significance.md (inside the harvest-committed
directory) plus stdout. Anything unreadable is reported as absent, never invented.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_gain_table import (ROOT, BENCHES, LABEL, pivot_cells,      # noqa: E402
                              authors_cells)
from build_missing_table import cell as missing_cell, RATES           # noqa: E402
from build_paper_table import sca_cell, authors_metrics, gram_cell    # noqa: E402

T_95_DF2 = 4.303   # two-sided 95% t quantile, n=3 -> df=2


def binom_tail(k, n):
    """One-sided exact binomial P(X >= k | n, p=0.5)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def seed_block(lines):
    lines.append('## 1. SCA seed statistics vs single-run baselines (two-stage T->V R@1)\n')
    lines.append('| benchmark | SCA mean +- sd (n=3) | 95% CI | GRAM* | PMRL* | HyperGRAM+ |')
    lines.append('|---|---|---|---|---|---|')
    for b in BENCHES:
        means, sds, seeds = sca_cell(b)
        if means is None:
            lines.append('| %s | (seed cells unreadable) | | | | |' % LABEL[b])
            continue
        m, sd = means[0], sds[0]
        half = T_95_DF2 * sd / math.sqrt(3)
        gram, _ = gram_cell(b)
        pmrl = authors_metrics(os.path.join(ROOT, 'workdir/pmrl_released', b, 'run.log'))
        hg = authors_metrics(os.path.join(ROOT, 'workdir/hgeval', b, 'run.log'))

        def mark(v):
            if v is None:
                return '--'
            out = '%.1f' % v[0] if isinstance(v, tuple) else '%.1f' % v
            x = v[0] if isinstance(v, tuple) else v
            return out + (' (outside CI)' if abs(x - m) > half else ' (inside CI)')
        lines.append('| %s | %.1f +- %.1f | [%.1f, %.1f] | %s | %s | %s |'
                     % (LABEL[b], m, sd, m - half, m + half,
                        mark(gram), mark(pmrl), mark(hg)))
    lines.append('')
    lines.append('Baselines are released checkpoints: n=1 by nature, so no two-sample test '
                 'exists. "outside CI" says the observed ordering is not explained by SCA '
                 'seed variance; it says nothing about the baseline\'s own variance.\n')


def sign_block(lines):
    lines.append('## 2. Exact sign tests\n')
    # Table 3: SCA vs GRAM on the masked aggregator
    wins = total = 0
    bench_wins = 0
    for b in BENCHES:
        bw = bt = 0
        for r in RATES:
            a = missing_cell('sca', b, r)
            g = missing_cell('gram', b, r)
            if a and g and a.get('agg') is not None and g.get('agg') is not None:
                bt += 1
                if a['agg'] > g['agg']:
                    bw += 1
        wins += bw
        total += bt
        if bt and bw == bt:
            bench_wins += 1
    if total:
        lines.append('- Masked-aggregator (Table 3): SCA beats GRAM in %d/%d cells; '
                     'exact one-sided sign test p = %.2e. Cells within a benchmark share '
                     'nested masks, so conservatively treating each BENCHMARK as one unit: '
                     '%d/%d, p = %.4f.' % (wins, total, binom_tail(wins, total),
                                           bench_wins, len(BENCHES),
                                           binom_tail(bench_wins, len(BENCHES))))
    # Gain table: negative baseline gains
    neg = tot = 0
    for name, cells in (('gram', pivot_cells('raw_vs_itm_missing.txt', 'gram')),
                        ('pmrl', authors_cells('pmrl_released.txt')),
                        ('hg', authors_cells('hypergram_authors.txt'))):
        for b, (best, agg) in cells.items():
            tot += 1
            if agg - best < 0:
                neg += 1
    if tot:
        lines.append('- Aggregation gain: %d/%d baseline method x benchmark cells negative; '
                     'against a fusion-helps-or-hurts-equally null, exact one-sided '
                     'p = %.4f. (Method rows share checkpoints across benchmarks -- the '
                     'cells are not fully independent; the count itself is the primary '
                     'evidence.)' % (neg, tot, binom_tail(neg, tot)))
    lines.append('')


def noise_block(lines):
    lines.append('## 3. Measured eval noise floor\n')
    path = os.path.join(ROOT, 'experiments/results/harvest/raw_vs_itm_frames.txt')
    spread = []
    if os.path.exists(path):
        for line in open(path):
            # repeat lines end with the spread column: "... 55.8 55.8      0.0"
            if 'ret_itm_area' in line:
                parts = line.split()
                try:
                    spread.append(float(parts[-1]))
                except ValueError:
                    pass
    if spread:
        lines.append('Repeat evaluations of identical checkpoints (harvest, %d cells): '
                     'max ITM R@1 spread %.1f. Margins at or below this are not results; '
                     'every headline margin in the paper is quoted against it.'
                     % (len(spread), max(spread)))
    else:
        lines.append('(no repeat-eval spreads found in the harvest)')
    lines.append('')


def main():
    lines = ['# Statistical support (generated by scripts/significance.py -- do not edit)\n']
    seed_block(lines)
    sign_block(lines)
    noise_block(lines)
    text = '\n'.join(lines)
    out = os.path.join(ROOT, 'experiments/results/tables_final/significance.md')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w').write(text + '\n')
    print(text)
    print('wrote experiments/results/tables_final/significance.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
