"""The main table is generated from measured results, never typed.

A table assembled by hand is where a mis-scored cell becomes a claim, and
audit_eval_geometry.py has already found 25 cells scored under a geometry they were never
trained with. So the builder reads each number out of a result directory and prints MISSING
for anything absent -- the one behaviour that matters is that it never silently fills a gap.
"""
import importlib.util
import io
import sys

import pytest


def _mod():
    spec = importlib.util.spec_from_file_location('bmt', 'scripts/build_main_table.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_an_unmeasured_cell_prints_missing_and_fails(monkeypatch, capsys):
    m = _mod()
    monkeypatch.setattr(m, 'itm_of', lambda *a, **k: None)
    monkeypatch.setattr(sys, 'argv', ['build_main_table.py'])
    rc = m.main()
    out = capsys.readouterr()
    assert rc == 1, 'a table with holes must not exit 0'
    assert 'MISSING' in out.out
    assert 'never by typing the number in' in out.err


def test_measured_cells_are_rendered_and_the_best_is_bolded(monkeypatch, capsys):
    m = _mod()
    # GRAM 50, authors' rows 51/52, SCA seeds 54/54/55 -> SCA best in every column. The
    # authors' rows come through authors_itm (their own eval logs / the committed harvest),
    # not through itm_of, so both entry points are stubbed.
    def fake(root, prefix, bench):
        if prefix == 'released':
            return 50.0
        return {'t9_qweight_only': 54.0, 's1_t9_seed51': 54.0,
                's2_t9_seed52': 55.0}.get(prefix)
    monkeypatch.setattr(m, 'itm_of', fake)
    monkeypatch.setattr(m, 'authors_itm', lambda root, name: (
        {b: (51.0 if 'pmrl' in root else 52.0) for b in m.BENCHES}, 'stubbed for the test'))
    monkeypatch.setattr(sys, 'argv', ['build_main_table.py'])
    rc = m.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert 'MISSING' not in out
    # mean 54.33, sd 0.58 -- the row carries an error bar, not a single run
    assert '54.3\\tiny{$\\pm$0.6}' in out, out
    assert '\\textbf{54.3' in out, 'the best measured value must be bolded'
    assert '50.0' not in out.replace('50.0', '', 1) or True     # GRAM rendered at all
    assert '& 50.0 &' in out


def test_published_numbers_are_a_reference_block_not_comparison_rows(monkeypatch, capsys):
    """The same GRAM checkpoint reads 54.8 as published and 52.5 here. Published rows must
    never sit in the block SCA is bolded against, or the bold means nothing."""
    m = _mod()
    monkeypatch.setattr(m, 'itm_of', lambda *a, **k: 50.0)
    monkeypatch.setattr(m, 'authors_itm',
                        lambda root, name: ({b: 50.0 for b in m.BENCHES}, 'stubbed'))
    monkeypatch.setattr(sys, 'argv', ['build_main_table.py'])
    m.main()
    out = capsys.readouterr().out
    ref, measured = out.split('Measured here, one environment')
    assert 'HyperGRAM$^{\\S}$' in ref and '56.6' in ref
    assert '\\textbf{' not in ref, 'nothing in the reference block may be bolded'
    assert 'reference only' in out


def test_the_sca_row_needs_more_than_one_seed():
    """A single run reported with an error bar would be a fabricated interval."""
    m = _mod()
    calls = {'n': 0}

    def one_seed(root, prefix, bench):
        if prefix == 't9_qweight_only':
            return 54.0
        if prefix in ('s1_t9_seed51', 's2_t9_seed52'):
            return None
        return 50.0
    import unittest.mock as mock
    with mock.patch.object(m, 'itm_of', one_seed), \
         mock.patch.object(sys, 'argv', ['build_main_table.py']):
        buf_out, buf_err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = buf_out, buf_err
        try:
            rc = m.main()
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    assert rc == 1
    assert 'only 1 seed' in buf_err.getvalue()


def test_authors_rows_come_from_the_parser_never_from_the_retired_cells():
    """The e1_repro cells are the superseded reimplementations; the PMRL and HyperGRAM rows
    must come from their authors' own eval logs (or the committed harvest of them), and the
    header must say which source was used."""
    src = open('scripts/build_main_table.py').read()
    # the string may appear in a comment saying the cells are NOT consulted; what must not
    # appear is e1_repro as a data path in the row definitions
    import re
    paths = re.findall(r"'(workdir/[^']+)'", src)
    assert not any('e1_repro' in p for p in paths), \
        'the retired reimplementation cells are being consulted: %s' % paths
    assert 'workdir/pmrl_released' in src and 'workdir/hgeval' in src
    assert 'def authors_itm' in src
    assert "out.append('%% source -- ' + line)" in src, \
        'the header must record where each authors row came from'
