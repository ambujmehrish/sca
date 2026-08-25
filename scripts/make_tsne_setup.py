#!/usr/bin/env python3
"""Setup for the latent-space (t-SNE) figure: a 3-class VGGSound subset + eval configs.

    python3 scripts/make_tsne_setup.py

Mirrors GRAM's Fig.-style visualization (a few acoustic classes, per-modality markers,
text stars) on OUR two models: SCA (T9) and the released GRAM checkpoint. Writes

  benchmark_eval/vgg_tsne_annotation.json     ~51 clips, 3 visually+acoustically distinct
                                              classes drawn from the VGGSound-5K annotation
  benchmark_eval/configs_tsne/{sca,gram}_vggtsne.json
                                              the validated vggsound configs with ONLY the
                                              annotation path swapped (drift-asserted)

The eval run (slurm_scripts/tsne_dump.sh) scores nothing new -- it exists to trigger the
SCA_DUMP_FEATS side channel in evaluation_classification.py, which saves the raw
per-modality unit vectors. scripts/plot_tsne.py renders the two panels from the dumps.
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# Eight acoustically distinctive classes (~17 clips each). The FIGURE displays only three,
# chosen by a deterministic rule in plot_tsne.py and disclosed in the caption; the numbers
# printed on the panels are computed over ALL eight, so the selection cannot bias them.
CLASSES = ('airplane flyby', 'bird squawking', 'car engine knocking',
           'car engine starting', 'cat purring', 'chainsawing trees',
           'chicken crowing', 'cow lowing')


def main():
    anno = json.load(open(os.path.join(ROOT, 'benchmark_eval/vgg5k_annotation_5000.json')))
    sub = [x for x in anno if x['desc'] in CLASSES]
    counts = {c: sum(1 for x in sub if x['desc'] == c) for c in CLASSES}
    assert all(v >= 10 for v in counts.values()), 'class too small: %s' % counts
    out_anno = os.path.join(ROOT, 'benchmark_eval/vgg_tsne_annotation.json')
    json.dump(sub, open(out_anno, 'w'))
    print('wrote %s (%d clips: %s)' % ('benchmark_eval/vgg_tsne_annotation.json',
                                       len(sub), counts))

    for model, tpl in (('sca', 'benchmark_eval/configs_qweight/sca_vggsound.json'),
                       ('gram', 'benchmark_eval/configs_e1/gram_vggsound.json')):
        base = json.load(open(os.path.join(ROOT, tpl)))
        cfg = json.loads(json.dumps(base))
        for blk in cfg['data_cfg']['train'], cfg['data_cfg']['val']:
            assert blk[0]['txt'] == 'benchmark_eval/vgg5k_annotation_5000.json'
            blk[0]['txt'] = 'benchmark_eval/vgg_tsne_annotation.json'
        out = os.path.join(ROOT, 'benchmark_eval/configs_tsne/%s_vggtsne.json' % model)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(cfg, open(out, 'w'), indent=1)
        a, b = json.load(open(os.path.join(ROOT, tpl))), json.load(open(out))
        for blk in ('train', 'val'):
            a['data_cfg'][blk][0].pop('txt'), b['data_cfg'][blk][0].pop('txt')
        assert a == b, 'non-annotation drift in %s' % out
        print('wrote benchmark_eval/configs_tsne/%s_vggtsne.json (drift-asserted)' % model)
    return 0


if __name__ == '__main__':
    sys.exit(main())
