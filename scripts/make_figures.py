#!/usr/bin/env python3
"""Paper figures from the committed E4/E5/E6 grid JSONs (no cluster access needed).

  python3 scripts/make_figures.py [--indir experiments/results/e4] [--outdir figures]

Fig 1  fig_e4_missingness   R@1 vs missing rate, mean +/- std over mask seeds.
Fig 2  fig_e6_calibration   slope (ref line at 1) and |R^2| distance-to-calibrated.

Color = method FAMILY (SCA blue / GRAM orange / PMRL aqua -- validated categorical
slots); variants within a family differ by line style and are direct-labeled, so
identity is never color-alone.
"""
import os
import json
import glob
import argparse
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e5e4dd'

# arm -> (label, native scorer, family color, linestyle)
ARMS = {
    'sca_t1':      ('SCA-T1 (ours)', 'centroid',      BLUE,   '-'),
    'sca':         ('SCA (ours)',    'centroid',      BLUE,   '--'),
    'sca_nomask':  ('SCA w/o masked training', 'centroid', BLUE, ':'),
    'gram':        ('GRAM',          'volume_masked', ORANGE, '-'),
    'gram_lora':   ('GRAM-LoRA',     'volume_masked', ORANGE, '--'),
    'pmrl_masked': ('PMRL-masked',   'pmrl_raw',      AQUA,   '-'),
}
RATES = ['0%|rand', '25%|rand', '50%|rand', '75%|rand', '90%|rand']
X = [0, 25, 50, 75, 90]


def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def load(indir, arm):
    return [json.load(open(p)) for p in sorted(glob.glob(f'{indir}/{arm}_s*.json'))]


def fig_e4(indir, outdir):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    _style(ax)
    ends = []
    for arm, (label, m, color, ls) in ARMS.items():
        seeds = load(indir, arm)
        if not seeds:
            continue
        mu = [st.mean([d['e4'][m][r]['R@1'] for d in seeds]) for r in RATES]
        sd = [st.stdev([d['e4'][m][r]['R@1'] for d in seeds]) if len(seeds) > 1 else 0
              for r in RATES]
        ax.plot(X, mu, ls, color=color, lw=2, solid_capstyle='round')
        ax.fill_between(X, [a - b for a, b in zip(mu, sd)],
                        [a + b for a, b in zip(mu, sd)], color=color, alpha=0.12, lw=0)
        ends.append([mu[-1], label])
    # spread end labels so converging lines (the finding itself) stay readable
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 1.15:
            ends[i][0] = ends[i - 1][0] + 1.15
    for ypos, label in ends:
        ax.annotate(label, (X[-1] + 2.0, ypos), color=INK, fontsize=8, va='center')
    ax.set_xlim(0, 122)
    ax.set_xticks(X)
    ax.set_xticklabels([f'{x}%' for x in X])
    ax.set_xlabel('missing-modality rate (test time)', color=INK, fontsize=9)
    ax.set_ylabel('R@1 (raw embedding space)', color=INK, fontsize=9)
    ax.set_title('Retrieval under missing modalities (MSR-VTT, mean ± std over 3 mask seeds)',
                 color=INK, fontsize=9.5, loc='left', pad=10)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outdir}/fig_e4_missingness.{ext}', bbox_inches='tight')
    plt.close(fig)


def fig_e6(indir, outdir):
    rows = []
    for arm, (label, m, color, _) in ARMS.items():
        p = sorted(glob.glob(f'{indir}/{arm}_s0.json'))
        if not p:
            continue
        o = json.load(open(p[0]))['e6'][m]['0%']['overall']
        rows.append((label, color, o['slope'], o['r2']))
    rows = rows[::-1]                                          # ours at top after invert
    y = range(len(rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9), dpi=200, sharey=True)
    for ax in (a1, a2):
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
    a1.axvline(1.0, color=INK, lw=1, ls='-', alpha=0.55, zorder=1)
    a1.annotate('calibrated\n(slope = 1)', (1.0, len(rows) - 0.4), color=MUTED,
                fontsize=7.5, ha='center')
    for i, (label, color, slope, r2) in enumerate(rows):
        a1.plot([slope], [i], 'o', color=color, ms=7)
        a1.plot([1.0, slope], [i, i], color=color, lw=1.2, alpha=0.5)
        a2.barh(i, abs(r2), color=color, height=0.55)
        a2.annotate(f'{r2:.2f}', (abs(r2) * 1.15, i), color=INK, fontsize=7.5,
                    va='center')
    a1.set_yticks(list(y))
    a1.set_yticklabels([r[0] for r in rows], fontsize=8, color=INK)
    a1.set_xlim(0.3, 3.1)
    a1.set_xlabel('regression slope S vs 2S*−1', color=INK, fontsize=8.5)
    a2.set_xscale('log')
    a2.set_xlim(0.3, 200)
    a2.set_xlabel('|R²|  (log scale — closer to 0 is better)', color=INK, fontsize=8.5)
    fig.suptitle('Semantic calibration (E6, known pairs): SCA scores track true similarity',
                 color=INK, fontsize=9.5, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outdir}/fig_e6_calibration.{ext}', bbox_inches='tight')
    plt.close(fig)


TARMS = {'sca_t1': ('SCA-T1 (ours)', 'centroid', BLUE, '-'),
         'gram_lora': ('GRAM-LoRA', 'volume_masked', ORANGE, '--'),
         'gram': ('GRAM', 'volume_masked', ORANGE, '-')}
BENCH = [('didemo', 'DiDeMo'), ('activitynet', 'ActivityNet'), ('audiocaps', 'AudioCaps')]


def fig_transfer(indir, outdir):
    """E4 grids OFF the selection benchmark: three panels, shared y-label."""
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), dpi=200, sharey=True)
    for ax, (key, title) in zip(axes, BENCH):
        _style(ax)
        for arm, (label, m, color, ls) in TARMS.items():
            seeds = [json.load(open(p))
                     for p in sorted(glob.glob(f'{indir}/{arm}_{key}_s*.json'))]
            if not seeds:
                continue
            mu = [st.mean([d['e4'][m][r]['R@1'] for d in seeds]) for r in RATES]
            sd = [st.stdev([d['e4'][m][r]['R@1'] for d in seeds]) if len(seeds) > 1 else 0
                  for r in RATES]
            ax.plot(X, mu, ls, color=color, lw=2, label=label, solid_capstyle='round')
            ax.fill_between(X, [a - b for a, b in zip(mu, sd)],
                            [a + b for a, b in zip(mu, sd)], color=color, alpha=0.12, lw=0)
        ax.set_xticks(X)
        ax.set_xticklabels([f'{x}%' for x in X], fontsize=7.5)
        ax.set_title(title, color=INK, fontsize=9, loc='left')
        ax.set_xlabel('missing rate', color=INK, fontsize=8.5)
    axes[0].set_ylabel('R@1 (raw embedding space)', color=INK, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=8, loc='upper right',
                   labelcolor=INK, handlelength=1.6)
    fig.suptitle('Off the selection benchmark: SCA beats GRAM at every rate, and overtakes '
                 'GRAM-LoRA as modalities disappear (mean ± std, 3 mask seeds)',
                 color=INK, fontsize=9.5, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outdir}/fig_e4_transfer.{ext}', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', default='experiments/results/e4')
    ap.add_argument('--outdir', default='figures')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fig_e4(a.indir, a.outdir)
    fig_e6(a.indir, a.outdir)
    tdir = a.indir.replace('/e4', '/e4_transfer')
    if glob.glob(f'{tdir}/*.json'):
        fig_transfer(tdir, a.outdir)
    print(f'-> {a.outdir}/fig_e4_missingness.(pdf|png), fig_e6_calibration.(pdf|png)')


