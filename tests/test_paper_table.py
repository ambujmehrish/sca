"""The per-benchmark paper tables (both directions, R@1/R@10, sectioned by geometry)."""
import subprocess

SRC = open('scripts/build_paper_table.py').read()


def test_row_provenance_is_the_project_policy():
    """GRAM and PMRL from released checkpoints, HyperGRAM from the authors' code (they
    release none), SCA as the single configuration over three seeds."""
    # the source spells the backslash escaped, so match the source text, not the LaTeX
    assert r"GRAM$^{\\star}$" in SRC
    assert r"PMRL$^{\\star}$" in SRC
    assert r"HyperGRAM$^{\\dagger}$" in SRC
    assert "SCA_SEEDS = ('t9_qweight_only', 's1_t9_seed51', 's2_t9_seed52')" in SRC


def test_hypergram_audiocaps_is_a_deliberate_dash_not_missing():
    """Their release does not run the audio-anchor benchmark and their paper reports no
    AudioCaps number -- a scope fact, stated in the caption, distinct from a hole in ours."""
    assert "'audiocaps')" in SRC and 'ABSENT' in SRC
    assert 'reports no AudioCaps number' in SRC
    r = subprocess.run(['python3', 'scripts/build_paper_table.py', '--bench', 'audiocaps'],
                       capture_output=True, text=True)
    hg = [l for l in r.stdout.splitlines() if l.startswith('HyperGRAM$^{\\dagger}$')][0]
    # the four METRIC cells must be the deliberate dash; the Params cell may read MISSING
    # until count_trainable has run, which is a different (and honest) state
    assert hg.rstrip('\\ ').endswith('-- & -- & -- & --'), hg


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


def test_table2_shares_the_extraction_with_the_per_benchmark_tables():
    """Two tables disagreeing about one number is the failure this import prevents."""
    src = open('scripts/build_transfer_table.py').read()
    assert 'from build_paper_table import' in src
    assert 'gram_cell' in src and 'sca_cell' in src and 'authors_metrics' in src
    for bad in ('def gram_cell', 'def sca_cell', 'def authors_metrics', 'def cell_metrics'):
        assert bad not in src, '%s reimplemented instead of imported' % bad


def test_table2_layout_and_policy():
    import subprocess
    r = subprocess.run(['python3', 'scripts/build_transfer_table.py',
                        '--out', '/tmp/claude-0/-home-user-sca/725c81e1-2702-55f2-8f1c-81a27a02a7ad/scratchpad/t2.tex'],
                       capture_output=True, text=True)
    out = open('/tmp/claude-0/-home-user-sca/725c81e1-2702-55f2-8f1c-81a27a02a7ad/scratchpad/t2.tex').read()
    for b in ('DiDeMo', 'ActivityNet', 'VATEX', 'AudioCaps'):
        assert '\\multicolumn{2}{c}{%s}' % b in out
    assert 'MSR-VTT' not in out, 'MSR-VTT belongs to Table 1'
    for line in out.splitlines():
        if '$^{\\S}$' in line:
            assert '\\textbf' not in line, 'published rows must never be bolded'
    # HyperGRAM/AudioCaps is a deliberate dash even when other cells are MISSING locally
    hg = [l for l in out.splitlines() if l.startswith('HyperGRAM')][0]
    assert hg.rstrip('\\ ').endswith('--'), hg


def test_missing_modality_sweep_is_paired_and_anchored():
    """The sweep's fairness rests on three properties: the gram configs inherit the exact
    Table-1 config plus only the two mask keys; both sides carry identical mask keys at
    every rate; and r00 exists as the masking-off control that must reproduce Table 1/2."""
    import json, subprocess
    r = subprocess.run(['python3', 'scripts/make_missing_configs.py'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'mask keys agree at every rate' in r.stdout
    base = json.load(open('benchmark_eval/configs_e1/gram_msrvtt.json'))['model_cfg']
    masked = json.load(open('benchmark_eval/configs_missing/r50/gram_msrvtt.json'))['model_cfg']
    extra = {k for k in masked if k not in base}
    assert extra == {'eval_mask_rate', 'eval_mask_seed'}, extra
    for k in base:
        assert masked[k] == base[k], '%s drifted from the Table-1 config' % k
    r00 = json.load(open('benchmark_eval/configs_missing/r00/gram_msrvtt.json'))['model_cfg']
    assert r00['eval_mask_rate'] == 0.0, 'the control anchor must be masking-off exactly'
    launch = open('slurm_scripts/missing_eval.sh').read()
    assert 'GRAM_RELEASED_CKPT' in launch and 'set GRAM_RELEASED_CKPT' in launch
    assert 't9_qweight_only/ckpt/model_step_5330.pt' in launch
    assert 'must reproduce Table 1/2' in launch
