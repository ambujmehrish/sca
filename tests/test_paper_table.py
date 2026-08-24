"""The per-benchmark paper tables (both directions, R@1/R@10, sectioned by geometry)."""
import subprocess

SRC = open('scripts/build_paper_table.py').read()


def test_row_provenance_is_the_project_policy():
    """GRAM and PMRL from released checkpoints, HyperGRAM from the authors' code (they
    release none), SCA as the single configuration over three seeds."""
    assert "GRAM$^{\\star}$ (released ckpt)" in SRC
    assert "PMRL$^{\\star}$ (released ckpt)" in SRC
    assert "HyperGRAM$^{\\dagger}$ (authors' code)" in SRC
    assert "SCA_SEEDS = ('t9_qweight_only', 's1_t9_seed51', 's2_t9_seed52')" in SRC


def test_hypergram_audiocaps_is_a_deliberate_dash_not_missing():
    """Their release does not run the audio-anchor benchmark and their paper reports no
    AudioCaps number -- a scope fact, stated in the caption, distinct from a hole in ours."""
    assert "'audiocaps')" in SRC and 'ABSENT' in SRC
    assert 'reports no AudioCaps number' in SRC
    r = subprocess.run(['python3', 'scripts/build_paper_table.py', '--bench', 'audiocaps'],
                       capture_output=True, text=True)
    hg = [l for l in r.stdout.splitlines() if l.startswith('HyperGRAM$^{\\dagger}$')][0]
    assert 'MISSING' not in hg and '--' in hg


def test_published_rows_are_reference_only_and_never_bolded():
    r = subprocess.run(['python3', 'scripts/build_paper_table.py', '--bench', 'msrvtt'],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if '$^{\\S}$' in line:
            assert '\\textbf' not in line, line


def test_a_cell_that_cannot_be_read_is_missing_and_the_exit_code_says_so():
    r = subprocess.run(['python3', 'scripts/build_paper_table.py', '--bench', 'msrvtt'],
                       capture_output=True, text=True)
    if 'MISSING' in r.stdout:
        assert r.returncode == 1
        assert 'never a remembered number' in r.stderr
