#!/usr/bin/env python3
"""The paper's results figure: aggregation gain vs test-time masking rate.

    python3 scripts/plot_gain_vs_mask.py

One figure, two panels (MSR-VTT, VATEX -- the two benchmarks where fusion carries real
signal). Three trajectories per panel, every point a committed harvest cell:

    SCA (query-weighted, masked training)   -- the reported model
    SCA w/o masked training views           -- the g11 ablation: same hue, dashed/open
    GRAM (released checkpoint)              -- the volume baseline

The y=0 line is the semantic divide: gain > 0 means the method's own fusion beats its own
best unimodal score; gain < 0 means fusion destroys information the encoders already had.
The figure compresses Tables gain/missing/ablation into one read: fusion that adds
information keeps adding it as modalities vanish, and both alternatives sink.

Data: experiments/results/harvest/raw_vs_itm_missing.txt (GAIN column), parsed
positionally; a missing cell is a hard error, never an interpolated point. Output:
experiments/results/tables_final/fig_gain_vs_mask.{pdf,png}.
"""
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_notes import METHOD                                   # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PIVOT = os.path.join(ROOT, 'experiments/results/harvest/raw_vs_itm_missing.txt')
OUT = os.path.join(ROOT, 'experiments/results/tables_final/fig_gain_vs_mask')

RATES = [0, 25, 50, 75, 90]
PANELS = [('msrvtt', 'MSR-VTT'), ('vatex', 'VATEX')]
# entity -> (cell prefix, display, color, linestyle, markerfacecolor)
BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
SERIES = [
    ('sca', '%s (ours)' % METHOD, BLUE, '-', BLUE),
    ('g11_train_nomask', '%s w/o masked training' % METHOD, BLUE, (0, (4, 2)), 'white'),
    ('gram', 'GRAM (released ckpt)', ORANGE, '-', ORANGE),
]
_LINE = re.compile(r'^(\S+)\s+((?:[-+]?\d+\.\d+\s+){5}[-+]?\d+\.\d+)\s*(?:<-.*)?$')


def load():
    rows = {}
    for line in open(PIVOT):
        m = _LINE.match(line.rstrip())
        if m:
            rows[m.group(1)] = [float(x) for x in m.group(2).split()]
    data = {}
    for bench, _ in PANELS:
        for prefix, _, _, _, _ in SERIES:
            pts = []
            for r in RATES:
                cell = '%s_%s_r%02d' % (prefix, bench, r)
                if cell not in rows:
                    sys.exit('FATAL: cell %s not in %s -- the figure plots measured points '
                             'only.' % (cell, os.path.relpath(PIVOT, ROOT)))
                pts.append(rows[cell][4])                       # GAIN column
            data[(prefix, bench)] = pts
    return data


def main():
    data = load()
    plt.rcParams.update({
        'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'text.color': INK, 'axes.labelcolor': INK,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), sharey=True)
    ylim = (-22, 7)
    for ax, (bench, title) in zip(axes, PANELS):
        # zero = "fusion neither adds nor destroys"; the region below is lightly shaded
        ax.axhspan(ylim[0], 0, color='#52514e', alpha=0.055, lw=0, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
        for prefix, label, color, ls, mfc in SERIES:
            y = data[(prefix, bench)]
            ax.plot(RATES, y, ls=ls, color=color, lw=2, marker='o', ms=4.5,
                    markerfacecolor=mfc, markeredgecolor=color, markeredgewidth=1.4,
                    zorder=3, solid_capstyle='round',
                    label=label if bench == PANELS[0][0] else None)
            # selective direct labels: the endpoint value only
            va = 'bottom' if y[-1] >= 0 else 'top'
            ax.annotate('%+.1f' % y[-1], (RATES[-1], y[-1]),
                        xytext=(4, 5 if va == 'bottom' else -5),
                        textcoords='offset points', ha='left', va=va,
                        fontsize=8, color=color, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=6)
        ax.set_xlabel('gallery clips missing one modality $r$ (%)')
        ax.set_xticks(RATES)
        ax.set_xlim(-4, 103)
        ax.set_ylim(*ylim)
        ax.grid(axis='y', color=MUTED, alpha=0.15, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=3)
    axes[0].set_ylabel(r'aggregation gain $\Delta_{\mathrm{agg}}$ (R@1)')
    # polarity labels on the left panel, in muted ink (text never wears series color)
    axes[0].text(2, 5.6, 'fusion adds information', fontsize=7.5, color=MUTED,
                 style='italic', va='top')
    axes[0].text(2, -20.9, 'fusion destroys information', fontsize=7.5, color=MUTED,
                 style='italic', va='bottom')
    axes[0].legend(loc='center left', bbox_to_anchor=(0.015, 0.30), frameon=False,
                   handlelength=2.4, borderaxespad=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT + '.pdf', bbox_inches='tight')
    fig.savefig(OUT + '.png', dpi=220, bbox_inches='tight')
    print('wrote %s.{pdf,png}' % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
