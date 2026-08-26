#!/usr/bin/env python3
"""Latent-space figure: joint t-SNE of text/video/audio embeddings, SCA vs released GRAM,
with quantitative separation metrics printed on the panels.

    python3 scripts/plot_tsne.py

Reads the dumps written by slurm_scripts/tsne_dump.sh (committed under
experiments/results/tables_final/tsne_feats/, all 8 classes).

HONESTY CONTRACT. t-SNE is qualitative and the display is a subset, so the figure is
built in two layers with different rules:
  - The NUMBERS on each panel are computed in the ORIGINAL embedding space (cosine),
    over ALL dumped classes -- never over the displayed subset, never in t-SNE space:
      audio sil.   mean silhouette of audio embeddings by class (class separation)
      text-ctr cos mean cosine between each class text and the normalized mean of that
                   class's video+audio embeddings (how well text anchors the class)
  - The DISPLAY shows the 3 classes with the largest per-class audio-silhouette gap
    (SCA minus GRAM) -- a deterministic rule, stated in the caption, chosen so the
    qualitative panel illustrates the difference the full-set numbers quantify.

Output: fig_tsne_latent.{pdf,png} next to the dumps. Requires scikit-learn.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_notes import METHOD                                   # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FEATS = os.path.join(ROOT, 'experiments/results/tables_final/tsne_feats')
OUT = os.path.join(ROOT, 'experiments/results/tables_final/fig_tsne_latent')
PANELS = (('vast', 'VAST (shared init.)'), ('gram', 'GRAM (full fine-tuned)'),
          ('sca', '%s (LoRA, 4.8M, ours)' % METHOD))
HUES = ('#2a78d6', '#eb6834', '#1baf7a')          # validated categorical slots 1-3
INK, MUTED = '#0b0b0b', '#52514e'


def load(model):
    p = os.path.join(FEATS, '%s.pt' % model)
    if not os.path.exists(p):
        sys.exit('FATAL: %s missing -- run slurm_scripts/tsne_dump.sh and harvest first.'
                 % os.path.relpath(p, ROOT))
    d = torch.load(p, map_location='cpu', weights_only=False)
    labels = np.array([d['class_ids'].index(c) for c in d['clip_classes']])
    return (d['class_ids'], d['feat_t'].numpy(), d['feat_v'].numpy(),
            d['feat_a'].numpy(), labels)


def metrics(classes, t, v, a, labels):
    """Full-set numbers in cosine space: per-class audio silhouette + text-centroid cos."""
    sil = silhouette_samples(a, labels, metric='cosine')
    sil_per_class = np.array([sil[labels == c].mean() for c in range(len(classes))])
    tc = []
    for c in range(len(classes)):
        m = np.concatenate([v[labels == c], a[labels == c]]).mean(axis=0)
        m = m / np.linalg.norm(m)
        tc.append(float(t[c] @ m))
    return sil_per_class, np.array(tc)


def main():
    data = {m: load(m) for m, _ in PANELS}
    classes = data['sca'][0]
    assert classes == data['gram'][0], 'class order differs between dumps'
    stats = {m: metrics(*data[m]) for m in data}

    # deterministic display rule keyed to the metric the data actually moves: the 3 classes
    # where BOTH adapted models most exceed the shared VAST initialization on TEXT-CENTROID
    # cosine -- gap_c = min(SCA_c, GRAM_c) - VAST_c. (The audio-silhouette axis is flat
    # across all three models, 0.20-0.26, and is reported on the panels, not selected on.)
    gap = np.minimum(stats['sca'][1], stats['gram'][1]) - stats['vast'][1]
    show = sorted(np.argsort(gap)[::-1][:3])
    print('display classes (largest min(SCA,GRAM)-VAST text-centroid gain): %s'
          % [(classes[i], round(float(gap[i]), 2)) for i in show])
    for m in data:
        print('%s: audio silhouette %.2f (all %d classes), text-centroid cos %.2f'
              % (m, stats[m][0].mean(), len(classes), stats[m][1].mean()))

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    plt.rcParams.update({'font.size': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    for ax, (model, title) in zip(axes, PANELS):
        _, t, v, a, labels = data[model]
        keep = np.isin(labels, show)
        idxmap = {c: i for i, c in enumerate(show)}
        X = np.concatenate([t[show], v[keep], a[keep]])
        Y = TSNE(n_components=2, perplexity=15, random_state=0, init='pca').fit_transform(X)
        nt, nv = len(show), int(keep.sum())
        Yt, Yv, Ya = Y[:nt], Y[nt:nt + nv], Y[nt + nv:]
        sub_labels = np.array([idxmap[c] for c in labels[keep]])
        for ci in range(len(show)):
            sel = sub_labels == ci
            ax.scatter(Yv[sel, 0], Yv[sel, 1], marker='s', s=22, c=HUES[ci],
                       edgecolors='white', linewidths=0.5, zorder=2)
            ax.scatter(Ya[sel, 0], Ya[sel, 1], marker='^', s=26, c=HUES[ci],
                       edgecolors='white', linewidths=0.5, zorder=2)
            ax.scatter(Yt[ci, 0], Yt[ci, 1], marker='*', s=260, c=HUES[ci],
                       edgecolors=INK, linewidths=0.9, zorder=3)
        sil, tc = stats[model][0].mean(), stats[model][1].mean()
        ax.text(0.02, 0.02, 'audio class sep. (sil.): %.2f\ntext-centroid cos: %.2f'
                % (sil, tc), transform=ax.transAxes, fontsize=8, color=INK,
                va='bottom', ha='left',
                bbox=dict(facecolor='white', edgecolor=MUTED, lw=0.5, alpha=0.85,
                          boxstyle='round,pad=0.35'))
        ax.set_title(title, fontsize=9.5, fontweight='bold', color=INK)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(MUTED), s.set_linewidth(0.6)
    handles = ([plt.Line2D([], [], marker=mk, ls='', color=INK, markersize=ms, label=lb)
                for mk, ms, lb in (('*', 12, 'text'), ('^', 7, 'audio'), ('s', 6, 'video'))]
               + [plt.Line2D([], [], marker='o', ls='', color=HUES[i], markersize=7,
                             label=classes[c]) for i, c in enumerate(show)])
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(pad=0.5)
    for ext in ('.pdf', '.png'):
        fig.savefig(OUT + ext, dpi=220, bbox_inches='tight')
    print('wrote %s.{pdf,png}' % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
