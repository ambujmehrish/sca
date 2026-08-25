#!/usr/bin/env python3
"""Latent-space figure: joint t-SNE of text/video/audio embeddings, SCA vs released GRAM.

    python3 scripts/plot_tsne.py

Reads the dumps written by slurm_scripts/tsne_dump.sh (committed under
experiments/results/tables_final/tsne_feats/), embeds each model's stacked
[text; video; audio] unit vectors with one t-SNE per panel (shared hyperparameters,
fixed seed), and renders two side-by-side panels: color = class, marker = modality,
text embeddings as large stars. Output: fig_tsne_latent.{pdf,png} next to the dumps.

Read the figure as: how tightly does each model cluster the modalities of one class
around its text anchor? Requires scikit-learn; run wherever the dumps are checked out.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FEATS = os.path.join(ROOT, 'experiments/results/tables_final/tsne_feats')
OUT = os.path.join(ROOT, 'experiments/results/tables_final/fig_tsne_latent')
PANELS = (('sca', 'SCA (ours)'), ('gram', 'GRAM (released ckpt)'))
HUES = ('#2a78d6', '#eb6834', '#1baf7a')          # validated categorical slots 1-3
INK, MUTED = '#0b0b0b', '#52514e'


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    plt.rcParams.update({'font.size': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    classes = None
    for ax, (model, title) in zip(axes, PANELS):
        p = os.path.join(FEATS, '%s.pt' % model)
        if not os.path.exists(p):
            sys.exit('FATAL: %s missing -- run slurm_scripts/tsne_dump.sh and harvest first.'
                     % os.path.relpath(p, ROOT))
        d = torch.load(p, map_location='cpu', weights_only=False)
        classes = d['class_ids']
        t, v, a = d['feat_t'], d['feat_v'], d['feat_a']
        clip_cls = [classes.index(c) for c in d['clip_classes']]
        X = torch.cat([t, v, a]).numpy()
        Y = TSNE(n_components=2, perplexity=15, random_state=0, init='pca').fit_transform(X)
        nt, nv = t.shape[0], v.shape[0]
        Yt, Yv, Ya = Y[:nt], Y[nt:nt + nv], Y[nt + nv:]
        for ci in range(len(classes)):
            idx = [i for i, c in enumerate(clip_cls) if c == ci]
            ax.scatter(Yv[idx, 0], Yv[idx, 1], marker='s', s=22, c=HUES[ci],
                       edgecolors='white', linewidths=0.5, zorder=2)
            ax.scatter(Ya[idx, 0], Ya[idx, 1], marker='^', s=26, c=HUES[ci],
                       edgecolors='white', linewidths=0.5, zorder=2)
            ax.scatter(Yt[ci, 0], Yt[ci, 1], marker='*', s=260, c=HUES[ci],
                       edgecolors=INK, linewidths=0.9, zorder=3)
        ax.set_title(title, fontsize=9.5, fontweight='bold', color=INK)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(MUTED), s.set_linewidth(0.6)
    # one legend row below both panels: modality markers in ink, classes as colored patches
    handles = ([plt.Line2D([], [], marker=m, ls='', color=INK, markersize=ms, label=lb)
                for m, ms, lb in (('*', 12, 'text'), ('^', 7, 'audio'), ('s', 6, 'video'))]
               + [plt.Line2D([], [], marker='o', ls='', color=HUES[i], markersize=7,
                             label=classes[i]) for i in range(len(classes))])
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(pad=0.5)
    for ext in ('.pdf', '.png'):
        fig.savefig(OUT + ext, dpi=220, bbox_inches='tight')
    print('wrote %s.{pdf,png}' % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
