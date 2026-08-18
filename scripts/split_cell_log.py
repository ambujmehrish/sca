#!/usr/bin/env python3
"""Split e1_zs_eval.sh slurm logs into one log file per grid cell.

The e1 job writes every cell's eval output into ONE slurm .out; SCA and GRAM cells on
the same benchmark share evaluation family names, so parsing the combined log would mix
models. This slices on the launcher's own '== [cell] START' markers.

  python3 scripts/split_cell_log.py <outdir> <slurm_log> [<slurm_log> ...]

Later logs overwrite earlier ones for the same cell (a rerun supersedes). Then extract:
  python3 scripts/extract_results.py --run "SCA zs DiDeMo=<outdir>/sca_didemo.txt" ...
"""
import os
import re
import sys

START = re.compile(r'^== \[(?P<cell>[\w-]+)\] START')
END = re.compile(r'^== \[')                       # next cell marker (START/OK/FAILED/skip)


def split(outdir, logs):
    os.makedirs(outdir, exist_ok=True)
    written = {}
    for path in logs:
        cell, buf = None, []
        def flush():
            if cell and buf:
                out = os.path.join(outdir, f'{cell}.txt')
                with open(out, 'w') as f:
                    f.writelines(buf)
                written[cell] = (out, len(buf))
        for line in open(path, errors='replace'):
            m = START.match(line)
            if m:
                flush()
                cell, buf = m.group('cell'), [line]
                continue
            if cell and END.match(line):
                flush()
                cell, buf = None, []
                continue
            if cell:
                buf.append(line)
        flush()
    for cell, (out, n) in sorted(written.items()):
        print(f'{cell:24s} -> {out}  ({n} lines)')
    if not written:
        sys.exit(f'no "== [cell] START" markers found in: {logs}')
    return written


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    split(sys.argv[1], sys.argv[2:])
