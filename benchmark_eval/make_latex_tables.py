#!/usr/bin/env python3
"""LaTeX tables for the paper (plan P5: "xlsx -> LaTeX tables").

Merges published baseline rows (benchmark_eval/published_rows.json -- GRAM auto-extracted,
others hand-filled or null) with MEASURED rows and emits one booktabs table per section.
A null/absent value renders as -- (never 0). Best measured+published value per column is
bolded.

Measured rows: a directory of JSON files, each
  {"method": "SCA", "section": "zeroshot_t2v", "rows": {"MSR-VTT|T-V": [R1, R10], ...}}
(write these from the eval logs / run_eval_grids outputs as runs finish).

  python3 benchmark_eval/make_latex_tables.py --out tables/           # published only
  python3 benchmark_eval/make_latex_tables.py --measured results/rows --out tables/
"""
import os
import sys
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from import_published_rows import load_published_rows

SECTIONS = {'zeroshot_t2v': 'Zero-shot text-to-video retrieval (R@1 / R@10)',
            'zeroshot_v2t': 'Zero-shot video-to-text retrieval (R@1 / R@10)',
            'zeroshot_audio': 'Zero-shot audio benchmarks (@1 / @10)',
            'finetune_t2v': 'Fine-tuned text-to-video retrieval (R@1 / R@10)',
            'finetune_v2t': 'Fine-tuned video-to-text retrieval (R@1 / R@10)'}


def _esc(s):
    return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def load_measured(measured_dir):
    rows = {}
    for p in sorted(glob.glob(os.path.join(measured_dir, '*.json'))):
        with open(p) as f:
            d = json.load(f)
        for key in ('method', 'section', 'rows'):
            if key not in d:
                raise KeyError(f'{p}: measured-row file missing "{key}"')
        if d['section'] not in SECTIONS:
            raise KeyError(f"{p}: unknown section {d['section']!r} "
                           f'(expected one of {sorted(SECTIONS)})')
        for k, v in d['rows'].items():
            if not (isinstance(v, list) and len(v) == 2):
                raise ValueError(f'{p}: row {k!r} must be [R@1, R@10], got {v!r}')
        rows.setdefault(d['section'], {})[d['method']] = d['rows']
    return rows


def section_table(section, title, published, measured):
    methods = list(measured.get(section, {}))                       # measured first (ours)
    methods += [m for m in published if not m.startswith('_')
                and any(v is not None for v in published[m].get(section, {}).values())]
    keys = []
    for m in methods:
        src = measured.get(section, {}).get(m) or published.get(m, {}).get(section, {})
        for k in src:
            if k not in keys:
                keys.append(k)
    if not methods or not keys:
        return None

    def cell(m, k):
        src = measured.get(section, {}).get(m)
        v = (src or {}).get(k)
        if v is None:
            v = published.get(m, {}).get(section, {}).get(k)
        return v

    # bold the best R@1 and R@10 per benchmark row
    best = {}
    for k in keys:
        vals = [(m, cell(m, k)) for m in methods]
        for j in (0, 1):
            have = [(m, v[j]) for m, v in vals if v is not None and v[j] is not None]
            if have:
                best[(k, j)] = max(x for _, x in have)

    lines = [r'\begin{table}[t]', r'\centering', r'\caption{%s}' % _esc(title),
             r'\label{tab:%s}' % section, r'\small',
             r'\begin{tabular}{l%s}' % ('c' * len(keys)), r'\toprule',
             'Method & ' + ' & '.join(_esc(k.replace('|', ' ')) for k in keys) + r' \\',
             r'\midrule']
    for m in methods:
        cells = []
        for k in keys:
            v = cell(m, k)
            if v is None:
                cells.append('--')
            else:
                parts = ['--' if x is None else
                         (r'\textbf{%.1f}' % x) if best.get((k, j)) == x else ('%.1f' % x)
                         for j, x in enumerate(v)]
                cells.append(' / '.join(parts))
        lines.append(_esc(m) + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--measured', help='directory of measured-row JSONs')
    ap.add_argument('--out', required=True, help='output directory for .tex files')
    args = ap.parse_args()
    published = load_published_rows()
    measured = load_measured(args.measured) if args.measured else {}
    os.makedirs(args.out, exist_ok=True)
    written = []
    for section, title in SECTIONS.items():
        tex = section_table(section, title, published, measured)
        if tex is None:
            continue
        path = os.path.join(args.out, f'{section}.tex')
        with open(path, 'w') as f:
            f.write(tex + '\n')
        written.append(path)
    print(f'{len(written)} tables -> {args.out}: '
          f'{[os.path.basename(p) for p in written]}')


if __name__ == '__main__':
    main()
