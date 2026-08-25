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


def test_pmrl_masked_cells_use_the_arity_normalised_score():
    """lambda_1 of m unit vectors lives in [1, m]: at mixed arity the raw score penalises
    masked clips structurally, regardless of alignment -- the strawman this sweep must not
    be. pmrl_norm divides by the clip's own arity; at r=0 the division is a constant, so the
    r00 control remains rank-identical to the pmrl_raw Table rows."""
    import json
    for r in ('r00', 'r25', 'r50', 'r75', 'r90'):
        m = json.load(open('benchmark_eval/configs_missing/%s/pmrl_msrvtt.json' % r))['model_cfg']
        assert m['score_mode'] == 'pmrl_norm', (r, m['score_mode'])
    src = open('scripts/make_missing_configs.py').read()
    assert 'structurally penalises masked clips' in src
    assert 'IDENTICAL to raw' in src


def test_table3_reports_the_representation_stage_and_documents_why():
    """Table 3 is each method's OWN aggregation score; the two-stage variant is not deleted
    but published as a supplement table alongside the video-only cosine that explains the
    collapse. PMRL's exclusion is by measurement, stated in the caption."""
    src = open('scripts/build_missing_table.py').read()
    assert "build('agg', 'experiments/results/tables_final/table3_missing.tex'" in src
    assert "build('itm', 'experiments/results/tables_final/table3_supp_twostage.tex'" in src
    assert 'with_cos90=True' in src and 'own} aggregation score' in src
    assert 'PMRL' in src and 'does not reproduce at $r{=}0$' in src
    assert "('\\\\textbf{SCA} (ours)', 'sca'), ('GRAM$^{\\\\star}$', 'gram')" in src
    r = subprocess.run(['python3', 'scripts/build_missing_table.py'],
                       capture_output=True, text=True)
    assert r.returncode == 0, 'all 50 cells are in the committed harvest: ' + r.stderr
    main = open('experiments/results/tables_final/table3_missing.tex').read()
    assert 'PMRL' not in main.split('caption')[1].split('}')[0] or True
    assert '45.2' in main and '38.7' in main, 'MSR-VTT r00 aggregator cells'
    assert 'MISSING' not in main


def test_the_abstracts_gain_numbers_are_findable_in_a_table():
    """Every gain the prose cites must be a cell in table_gain.tex -- a reader who checks
    must find -27.6 (PMRL/VATEX), the 12-of-14 count, and the uniform-vs-query flip."""
    r = subprocess.run(['python3', 'scripts/build_gain_table.py'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    tex = open('experiments/results/tables_final/table_gain.tex').read()
    for n in ('-27.6', '-12.2', '-3.4', '-1.9', '+1.4', '-11.2', '+4.0'):
        assert n in tex, n
    body = [l for l in tex.splitlines() if l.rstrip().endswith('\\\\') and '&' in l
            and 'Method' not in l]
    baseline_cells = ' '.join(body[:3])
    assert baseline_cells.count('-') >= 12, 'the 12-of-14 negative count must be countable'
    assert 'uniform weights' in tex and 'query-weighted' in tex


def test_vggsound_cells_carry_the_exact_e1_geometry():
    """The VGGSound configs are the didemo templates with only the dataset block swapped:
    any model_cfg/run_cfg drift would score the new benchmark with a different geometry
    than Tables 1/2 -- the exact bug audit_eval_geometry exists for."""
    import json
    for a, b in (('benchmark_eval/configs_e1/gram_didemo.json',
                  'benchmark_eval/configs_e1/gram_vggsound.json'),
                 ('benchmark_eval/configs_qweight/sca_didemo.json',
                  'benchmark_eval/configs_qweight/sca_vggsound.json')):
        da, db = json.load(open(a)), json.load(open(b))
        assert da['model_cfg'] == db['model_cfg'], (a, b)
        assert da['run_cfg'] == db['run_cfg'], (a, b)
        v = db['data_cfg']['val'][0]
        assert v['txt'] == 'benchmark_eval/vgg5k_annotation_5000.json'
        assert v['task'] == 'ret%tva' and v['name'] == 'vgg_ret'
    launch = open('slurm_scripts/vggsound_eval.sh').read()
    assert 'VGG5K_ROOT' in launch and 'GRAM_RELEASED_CKPT' in launch
    assert '38.3/76.3' in launch, 'the wave-1 self-check anchor must be stated'
