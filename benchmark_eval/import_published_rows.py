#!/usr/bin/env python3
"""Published baseline rows for the results pipeline (plan P3 / §4).

Source of truth is benchmark_eval/published_rows.json. The GRAM rows are extracted
PROGRAMMATICALLY from make_results_xlsx.py's own "Paper" columns (no hand transcription);
the other §4 baselines (VAST, ImageBind, LanguageBind, UMT-L, InternVideo2, mPLUG-2,
VideoPrism) ship as EXPLICIT null slots -- a null renders as a dash in any table build and
is never a zero. Fill them from the respective papers; loaders refuse rows whose value
lists are the wrong length.

  python3 benchmark_eval/import_published_rows.py --regen    # rebuild GRAM rows from
                                                             # make_results_xlsx.py
  python3 benchmark_eval/import_published_rows.py --check    # verify json is in sync
"""
import os
import re
import ast
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX_SRC = os.path.join(HERE, '..', 'make_results_xlsx.py')
OUT = os.path.join(HERE, 'published_rows.json')

OTHER_BASELINES = ['VAST', 'ImageBind', 'LanguageBind', 'UMT-L', 'InternVideo-L',
                   'mPLUG-2', 'VideoPrism-b']


def _extract_list(src, name):
    m = re.search(rf'^{name}\s*=\s*(\[.*?^\])', src, re.M | re.S)
    assert m, f'{name} not found in make_results_xlsx.py'
    return ast.literal_eval(m.group(1))


def gram_rows_from_xlsx_source():
    src = open(XLSX_SRC).read()
    zs_t2v = _extract_list(src, 'zs_t2v')
    zs_v2t = _extract_list(src, 'zs_v2t')
    aud = _extract_list(src, 'aud')
    ft = _extract_list(src, 'ft')
    rows = {'zeroshot_t2v': {}, 'zeroshot_v2t': {}, 'zeroshot_audio': {},
            'finetune_t2v': {}, 'finetune_v2t': {}}
    for r in zs_t2v:                      # [bench, mode, ours R1, R10, base R1, R10, PAPER R1, R10, d]
        rows['zeroshot_t2v'][f'{r[0]}|{r[1]}'] = [r[6], r[7]]
    for r in zs_v2t:
        rows['zeroshot_v2t'][f'{r[0]}|{r[1]}'] = [r[6], r[7]]
    for r in aud:                         # [bench, metric, ours 1,10, base 1,10, PAPER 1,10, d]
        rows['zeroshot_audio'][r[0]] = [r[6], r[7]]
    for r in ft:                          # [bench, mode, t2v ours 1,10, t2v PAPER 1,10, v2t ours 1,10, v2t PAPER 1,10, d]
        rows['finetune_t2v'][f'{r[0]}|{r[1]}'] = [r[4], r[5]]
        rows['finetune_v2t'][f'{r[0]}|{r[1]}'] = [r[8], r[9]]
    return rows


def build():
    data = {
        '_meta': {
            'metric': '[R@1, R@10] (VGGSound: [Acc@1, Acc@10])',
            'note': ('GRAM (paper) rows auto-extracted from make_results_xlsx.py "Paper" '
                     'columns. null = not yet transcribed from the paper; renderers must '
                     'show a dash, NEVER 0. Regenerate GRAM rows with --regen.')},
        'GRAM (paper)': gram_rows_from_xlsx_source(),
    }
    gram_keys = data['GRAM (paper)']
    for b in OTHER_BASELINES:
        data[b] = {sec: {k: None for k in keys} for sec, keys in gram_keys.items()}
    return data


def load_published_rows(path=OUT):
    with open(path) as f:
        data = json.load(f)
    for method, sections in data.items():
        if method.startswith('_'):
            continue
        for sec, entries in sections.items():
            for key, val in entries.items():
                if val is not None and (not isinstance(val, list) or len(val) != 2):
                    raise ValueError(f'published_rows.json: {method}/{sec}/{key} must be '
                                     f'null or [R@1, R@10], got {val!r}')
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regen', action='store_true')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.regen:
        data = build()
        if os.path.exists(OUT):            # keep any hand-filled non-GRAM rows
            old = load_published_rows()
            for b in OTHER_BASELINES:
                if b in old:
                    data[b] = old[b]
        with open(OUT, 'w') as f:
            json.dump(data, f, indent=1)
        print(f'wrote {OUT}')
    elif args.check:
        cur = load_published_rows()
        fresh = gram_rows_from_xlsx_source()
        assert cur['GRAM (paper)'] == fresh, \
            'published_rows.json GRAM rows out of sync with make_results_xlsx.py -- run --regen'
        print('published_rows.json in sync '
              f'({sum(len(v) for v in fresh.values())} GRAM rows)')
    else:
        ap.error('pass --regen or --check')


if __name__ == '__main__':
    main()
