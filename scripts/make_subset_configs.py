#!/usr/bin/env python3
"""Modality-subset ladder configs (appendix): T-V and T-VA rungs on the tvas benchmarks.

    python3 scripts/make_subset_configs.py

The ladder evaluates the SAME checkpoints with a reduced modality set -- ret%tv and
ret%tva instead of the canonical ret%tvas -- on the two benchmarks that have three
modalities (MSR-VTT, VATEX). The top rung {V,A,S} is NOT regenerated: it is the already
measured canonical cell (Tables 1/2), so the ladder shares its endpoint with the main
tables by construction.

Each config is the validated per-benchmark template (configs_qweight/sca_*, configs_e1/
gram_*) with ONLY the val task string changed -- asserted, not assumed: any other drift
would score the ladder with a different geometry than the tables it extends. The task
string genuinely selects the volume's/centroid's modality set in our trunk (see
evaluation_mm.py:262 -- before that fix, tv/tva/tvas all scored the same full-arity
volume, which is why this must be generated on top of the fixed trunk, never the forks).
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BENCHES = ('msrvtt', 'vatex')
SUBSETS = {'tv': 'ret%tv', 'tva': 'ret%tva'}
TEMPLATE = {'sca': 'benchmark_eval/configs_qweight/sca_%s.json',
            'gram': 'benchmark_eval/configs_e1/gram_%s.json'}


def main():
    made = 0
    for model, tpl in TEMPLATE.items():
        for bench in BENCHES:
            src_path = os.path.join(ROOT, tpl % bench)
            base = json.load(open(src_path))
            assert base['data_cfg']['val'][0]['task'] == 'ret%tvas', \
                '%s is not a tvas benchmark config' % src_path
            for sub, task in SUBSETS.items():
                cfg = json.loads(json.dumps(base))            # deep copy
                for block in cfg['data_cfg']['train'], cfg['data_cfg']['val']:
                    block[0]['task'] = task
                out = os.path.join(ROOT, 'benchmark_eval/configs_subsets/%s/%s_%s.json'
                                   % (sub, model, bench))
                os.makedirs(os.path.dirname(out), exist_ok=True)
                json.dump(cfg, open(out, 'w'), indent=1)
                # drift assert: task is the ONLY difference
                a, b = json.load(open(src_path)), json.load(open(out))
                for blk in ('train', 'val'):
                    a['data_cfg'][blk][0].pop('task'), b['data_cfg'][blk][0].pop('task')
                assert a == b, 'non-task drift in %s' % out
                made += 1
    print('wrote %d subset configs; only the task string differs from the validated '
          'templates (asserted)' % made)
    return 0


if __name__ == '__main__':
    sys.exit(main())
