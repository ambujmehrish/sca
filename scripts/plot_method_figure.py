#!/usr/bin/env python3
"""The method figure, generated -- so it cannot drift from the implementation.

    python3 scripts/plot_method_figure.py

Every label is traced to code, not to an earlier design of the method:
  encoders      config/sca/default_model_cfg.json: vision_encoder_type=evaclip01_giant,
                audio_encoder_type=beats; subtitles and the text query share the BERT
                multimodal encoder (there is no separate subtitle encoder)
  adaptation    utils/build_optimizer.py: backbones frozen, LoRA r=8 (alpha=16) on q,v
  weights       model/centroid.py::query_weights, tau_w=0.1
  centroid      model/centroid.py::masked_spherical_mean (weighted, renormalised)
  score         cosine <z^T, mu>, model/centroid.py::query_centroid_scores
  masking       data/mask_sampler.py: p_full annealed 1 -> 0.5 over 2000 steps, one
                modality dropped uniformly, never below two present
  losses        model/losses_sca.py + model/sca.py; L_DAM is GRAM's data-anchor matching
                loss retained at 0.1 (GRAM Eq. 8 / HyperGRAM Eq. 11)

Output: experiments/results/tables_final/fig_method.{pdf,png}, sized for a two-column
figure* (7.0in wide).
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, FancyArrowPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'experiments/results/tables_final/fig_method')

V, A, S, T = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'      # validated palette slots
INK, MUTED, LINE = '#0b0b0b', '#52514e', '#9a9994'
PANEL_BG, BOX_BG = '#fbfbfa', '#ffffff'

FS_TITLE, FS_BODY, FS_MATH, FS_SMALL = 10.0, 8.6, 8.6, 7.4


def box(ax, x, y, w, h, text, ec=LINE, fc=BOX_BG, fs=FS_BODY, color=INK, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.02',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            color=color, zorder=3, linespacing=1.5)


def arrow(ax, p, q, color=MUTED, lw=1.1, style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, color=color,
                                 lw=lw, linestyle=ls, shrinkA=1.5, shrinkB=1.5, zorder=4))


def panel_title(ax, n, text):
    ax.text(0.035, 0.955, r'$\bf{(%s)}$  %s' % (n, text), ha='left', va='center',
            fontsize=FS_TITLE, color=INK)


# ----------------------------------------------------------------- panel A: encoding
def panel_encoding(ax):
    panel_title(ax, 'a', 'Encoding')
    rows = [('Video',  V, 'EVA-CLIP-g/14', r'$z^{V}$'),
            ('Audio',  A, 'BEATs',         r'$z^{A}$'),
            ('Subtitle', S, 'BERT',        r'$z^{S}$'),
            ('Query',  T, 'BERT',          r'$z^{T}$')]
    y0, dy, h = 0.790, 0.155, 0.105
    for i, (name, col, enc, sym) in enumerate(rows):
        y = y0 - i * dy
        box(ax, 0.04, y, 0.20, h, name, ec=col, fs=FS_SMALL, color=col)
        arrow(ax, (0.248, y + h / 2), (0.292, y + h / 2))
        box(ax, 0.30, y, 0.40, h, enc, ec=MUTED, fs=FS_SMALL)
        arrow(ax, (0.708, y + h / 2), (0.775, y + h / 2), color=col)
        ax.text(0.85, y + h / 2, sym, ha='center', va='center', fontsize=FS_MATH, color=col)
    ax.plot([0.04, 0.95], [0.455, 0.455], color=LINE, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(0.495, 0.255, 'query side: never aggregated', ha='center', va='center',
            fontsize=FS_SMALL, color=MUTED, style='italic')
    box(ax, 0.04, 0.035, 0.91, 0.125,
        'VAST backbone frozen\n' r'LoRA $r{=}8$ ($\alpha{=}16$) on $q,v$',
        ec=MUTED, fc=PANEL_BG, fs=FS_SMALL)


# ------------------------------------------------- panel B: query-conditioned centroid
def panel_aggregation(ax):
    panel_title(ax, 'b', 'Query-conditioned aggregation')
    cx, cy, r = 0.335, 0.585, 0.205
    ax.add_patch(Circle((cx, cy), r, fc='#f4f4f2', ec=LINE, lw=0.9, zorder=1))
    for w in (0.42, 0.80):
        ax.add_patch(Ellipse((cx, cy), 2 * r * w, 2 * r, fc='none', ec='#e4e4e0',
                             lw=0.7, zorder=1))
    ax.add_patch(Ellipse((cx, cy), 2 * r, 2 * r * 0.34, fc='none', ec='#e4e4e0',
                         lw=0.7, zorder=1))

    vecs = [(128, V, r'$z^{V}$', 0.44), (46, A, r'$z^{A}$', 0.36), (250, S, r'$z^{S}$', 0.20)]
    for ang, col, sym, _w in vecs:
        th = np.deg2rad(ang)
        px, py = cx + r * np.cos(th), cy + r * np.sin(th)
        arrow(ax, (cx, cy), (px, py), color=col, lw=1.4)
        ax.scatter([px], [py], s=22, color=col, zorder=5, edgecolors='white', linewidths=0.7)
        ax.text(cx + (r + 0.052) * np.cos(th), cy + (r + 0.052) * np.sin(th), sym,
                ha='center', va='center', fontsize=FS_MATH, color=col)

    ang_mu = np.rad2deg(np.arctan2(sum(w * np.sin(np.deg2rad(a)) for a, _, _, w in vecs),
                                   sum(w * np.cos(np.deg2rad(a)) for a, _, _, w in vecs)))
    th = np.deg2rad(ang_mu)
    arrow(ax, (cx, cy), (cx + r * np.cos(th), cy + r * np.sin(th)), color=INK, lw=2.1)
    ax.text(cx + 0.10, cy + r + 0.055, r'$\mu_{\mathcal{M}}(z^{T})$', ha='center',
            va='center', fontsize=FS_MATH, color=INK)

    # query star: bottom-left, clear of the sphere and of the score box
    qx, qy = 0.115, 0.295
    ax.scatter([qx], [qy], marker='*', s=170, color=T, edgecolors=INK, linewidths=0.7,
               zorder=6)
    ax.text(qx - 0.058, qy, r'$z^{T}$', ha='center', va='center', fontsize=FS_MATH, color=T)
    for ang, col, _s, _w in vecs:
        th2 = np.deg2rad(ang)
        arrow(ax, (qx, qy), (cx + r * np.cos(th2), cy + r * np.sin(th2)),
              color=T, lw=0.6, style='-', ls=(0, (2, 2)))

    # weight bars on the right: the query decides the mixture
    bx, by, bh = 0.635, 0.545, 0.058
    ax.text(bx, by + 3 * bh + 0.038, r'$w_{m}\propto e^{\langle z^{T}\!,z^{m}\rangle/\tau_{w}}$',
            ha='left', va='center', fontsize=FS_MATH, color=INK)
    for i, (_a, col, sym, w) in enumerate(vecs):
        y = by + (2 - i) * bh
        ax.add_patch(FancyBboxPatch((bx + 0.045, y), 0.28 * w / 0.44, bh * 0.58,
                                    boxstyle='round,pad=0.002,rounding_size=0.008',
                                    fc=col, ec='none', zorder=3))
        ax.text(bx + 0.035, y + bh * 0.29, sym, ha='right', va='center',
                fontsize=FS_SMALL, color=col)
    ax.text(bx, by - 0.048, r'$\tau_{w}=0.1$', ha='left', va='center',
            fontsize=FS_SMALL, color=MUTED)

    box(ax, 0.04, 0.115, 0.92, 0.115,
        r'$s=\langle z^{T}\!,\mu_{\mathcal{M}}(z^{T})\rangle\in[-1,1]$'
        r'   for any  $\mathcal{M}\subseteq\{V,A,S\}$',
        ec=MUTED, fc=PANEL_BG, fs=FS_MATH)
    ax.text(0.5, 0.055, 'a missing modality is one fewer term', ha='center', va='center',
            fontsize=FS_SMALL, color=MUTED, style='italic')


# ------------------------------------------------------------------ panel C: training
def panel_training(ax):
    panel_title(ax, 'c', 'Training')
    box(ax, 0.04, 0.795, 0.42, 0.10, r'complete $\mathcal{K}$', ec=MUTED, fs=FS_SMALL)
    box(ax, 0.54, 0.795, 0.42, 0.10,
        r'reduced $\mathcal{K}\!\setminus\!\{m^{\dagger}\}$', ec=MUTED, fs=FS_SMALL)
    arrow(ax, (0.468, 0.845), (0.532, 0.845), color=MUTED, style='<|-|>', lw=1.0)
    ax.text(0.5, 0.742, r'one forward pass  ·  $p_{\mathrm{full}}\!:\!1\!\to\!0.5$ (2k steps)',
            ha='center', va='center', fontsize=FS_SMALL, color=MUTED)

    losses = [
        (r'$\mathcal{L}_{\mathrm{align}}$', 'InfoNCE, both directions', V),
        (r'$\mathcal{L}_{\mathrm{mask}}$',
         r'$1-\langle\mu_{\mathcal{M}},\mu_{\mathcal{K}}\rangle$ '
         r'$+\,(s_{\mathcal{M}}-\mathrm{sg}[s_{\mathcal{K}}])^{2}$', A),
        (r'$\mathcal{L}_{\mathrm{sem}}$', r'graded targets $S^{*}$', S),
        (r'$\mathcal{L}_{\mathrm{unif}}$', r'spread $\mu_{i}$ on the sphere', T),
    ]
    y0, h, gap = 0.598, 0.093, 0.026
    for i, (name, desc, col) in enumerate(losses):
        y = y0 - i * (h + gap)
        ax.add_patch(FancyBboxPatch((0.04, y), 0.92, h,
                                    boxstyle='round,pad=0.005,rounding_size=0.018',
                                    fc=BOX_BG, ec=col, lw=1.0, zorder=2))
        ax.text(0.075, y + h / 2, name, ha='left', va='center', fontsize=FS_MATH,
                color=col, zorder=3)
        ax.text(0.235, y + h / 2, desc, ha='left', va='center', fontsize=FS_SMALL,
                color=INK, zorder=3)
    box(ax, 0.04, 0.035, 0.92, 0.145,
        r'$\mathcal{L}=\mathcal{L}_{\mathrm{align}}+\mathcal{L}_{\mathrm{sem}}'
        r'+\mathcal{L}_{\mathrm{mask}}$' '\n'
        r'$+\,0.1\,\mathcal{L}_{\mathrm{unif}}+0.1\,\mathcal{L}_{\mathrm{DAM}}$',
        ec=INK, fc=PANEL_BG, fs=FS_MATH, lw=1.2)


def main():
    plt.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    fig = plt.figure(figsize=(7.0, 3.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.80, 1.22, 1.05], wspace=0.055,
                          left=0.004, right=0.996, top=0.996, bottom=0.004)
    for i, fn in enumerate((panel_encoding, panel_aggregation, panel_training)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_facecolor('#ffffff')
        for s in ax.spines.values():
            s.set_color(LINE), s.set_linewidth(0.8)
        fn(ax)
    for ext in ('.pdf', '.png'):
        fig.savefig(OUT + ext, dpi=400, bbox_inches='tight', pad_inches=0.02)
    print('wrote %s.{pdf,png}' % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
