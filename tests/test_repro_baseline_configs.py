"""Same-environment baselines: only the aggregator differs.

The published table mixes four evaluation environments, and we have measured that they
differ -- the same released GRAM checkpoint reads 54.8 in the paper and 52.5 here, 83.5 there
and 90.0 here on VATEX. Differences of one or two points across environments are therefore
uninterpretable, which is most of the range this field competes in.

These configs remove that variable: same foundation checkpoint, same 150k training set, same
eval data blocks, same frames and rerank depth, with only the scoring geometry changed. A
difference in that table is a difference in the aggregator, which is the algorithmic claim.
The tests below pin the "only the aggregator differs" property, since it is the whole point
and it is invisible once the numbers are in a table.
"""
import glob
import json
import os

import pytest

BENCHES = ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps')
METHODS = ('pmrl', 'hypergram', 'gram_lora')


def _repro(method, bench):
    return json.load(open('benchmark_eval/configs_repro/%s_%s.json' % (method, bench)))


def _sca(bench):
    return json.load(open('benchmark_eval/configs_qweight/sca_%s.json' % bench))


def test_all_fifteen_cells_exist():
    got = sorted(os.path.basename(p) for p in glob.glob('benchmark_eval/configs_repro/*.json'))
    assert len(got) == len(METHODS) * len(BENCHES), got


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('bench', BENCHES)
def test_the_protocol_is_identical_to_the_sca_row(method, bench):
    """Data, frames, batch, task and rerank depth are held fixed. If any of these drifted, a
    difference in the table would be a difference in the protocol wearing an aggregator's
    name."""
    r, s = _repro(method, bench), _sca(bench)
    assert r['data_cfg'] == s['data_cfg'], 'data block differs'
    # compare with .get on both sides: some benchmarks set max_caption_len explicitly and
    # others inherit it from default_model_cfg, so requiring the key would fail on the
    # inheriting ones for a reason that has nothing to do with the protocol matching.
    for k in ('itm_rerank_num', 'max_caption_len', 'vision_encoder_type',
              'ret_bidirection_evaluation'):
        assert r['model_cfg'].get(k) == s['model_cfg'].get(k), \
            '%s differs: repro=%r sca=%r' % (k, r['model_cfg'].get(k), s['model_cfg'].get(k))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('bench', BENCHES)
def test_no_baseline_carries_sca_specific_machinery(method, bench):
    """A baseline scored with the query-weighted centroid is not that baseline. s_star_path is
    SCA's semantic-target cache and has no meaning for any of these."""
    m = _repro(method, bench)['model_cfg']
    for k in ('sca_query_weighting', 'sca_tau_w', 'sca_frame_slots', 's_star_path'):
        assert k not in m, '%s/%s still carries %s' % (method, bench, k)


@pytest.mark.parametrize('method', METHODS)
def test_each_method_declares_the_geometry_that_defines_it(method):
    m = _repro(method, 'msrvtt')['model_cfg']
    # score_mode is now STATED per method rather than inherited -- 'volume' is GRAM's
    # geometry and used to be left implicit, which is exactly how it resolved to centroid
    expect = {'pmrl': ('pmrl', 'pmrl_raw'),
              'hypergram': ('gram_hyp', 'volume'),
              'gram_lora': ('gram_lora', 'volume')}[method]
    assert m.get('model_type') == expect[0], m.get('model_type')
    assert m.get('score_mode') == expect[1], m.get('score_mode')


def test_the_launcher_refuses_a_config_that_does_not_match_the_checkpoint():
    """Scoring a PMRL checkpoint through the volume, or a hypergraph checkpoint through the
    centroid, yields a full set of plausible numbers for a model that was never trained: no
    shape mismatch, no error, and a table wrong in a way nobody can see."""
    src = open('slurm_scripts/repro_baselines_eval.sh').read()
    assert 'trained with' in src, 'no cross-check against the arm hps.json'
    assert "'model_type', 'score_mode'" in src, 'the cross-check must cover both'


def test_the_hypergram_caveat_travels_with_the_launcher():
    """The 37.4 figure has been quoted repeatedly as "our HyperGRAM reproduction does not
    work". It came from gram_hyp2, trained at lr 2e-5 -- the rate wave4/ANALYSIS.md had already
    identified as the recipe defect for our OWN method (SCA went 53.5 -> 54.9 when moved to
    1e-4). It is a run at a rate the method's paper does not use, in a family already shown to
    be learning-rate sensitive, so it is not evidence about HyperGRAM.

    The launcher must point at HYPERGRAM_STATUS.md and must not present 37.4 as a reproduction
    result, because this conclusion has been re-derived from the stale number more than once."""
    src = open('slurm_scripts/repro_baselines_eval.sh').read()
    assert 'HYPERGRAM_STATUS.md' in src, 'the launcher must point at the status file'
    assert 'NOT evidence' in src, 'the stale number must be marked as not evidence'
    assert 'never HyperGRAM' in src
    assert 'We do not reproduce their result' not in src, \
        'that phrasing asserts a reproduction failure the runs do not support'


def test_the_status_file_records_the_authors_actual_recipe():
    """Their code is public (github.com/uta-smile/HyperGram), so the recipe is not a matter of
    reading their paper: lr 5e-5, ONE epoch, task ret%tvas%tv%ta, and a curvature parameter
    group at 10x the base lr. Every arm we ran missed all three of the first ones. The status
    file has to carry those values, because the wrong ones were re-derived more than once."""
    txt = open('experiments/results/HYPERGRAM_STATUS.md').read()
    assert 'uta-smile/HyperGram' in txt, 'the released code must be named'
    assert '5e-05' in txt or '5e-5' in txt, "their learning rate must be recorded"
    assert 'ret%tvas%tv%ta' in txt, 'their task mix includes subtitles and ours does not'
    assert '10x the base learning rate' in txt, 'the curvature lr group is not guessable'
    assert 'must never be cited' in txt, 'the stale 37.4 must be marked unusable'


def test_no_shipped_config_claims_to_be_hypergrams_recipe():
    """H1 was named _paper while matching GRAM's recipe rather than HyperGRAM's. A config that
    claims a paper's setup and does not have it is worse than an unnamed one."""
    import glob
    for p in glob.glob('config/sca/ablations/*hypergram*.json'):
        c = json.load(open(p))
        lr = c['run_cfg'].get('learning_rate')
        ep = c['data_cfg']['train'][0].get('epoch')
        task = c['data_cfg']['train'][0].get('task')
        assert (lr, ep, task) == (5e-05, 1, 'ret%tvas%tv%ta'), (
            '%s claims HyperGRAM but has lr=%s epoch=%s task=%s; theirs is 5e-05 / 1 / '
            'ret%%tvas%%tv%%ta' % (p, lr, ep, task))


def test_the_wrong_recipe_arm_is_gone():
    """H1 was gram_hyp2 at GRAM's learning rate, built while we believed HyperGRAM's code was
    unavailable. It is: lr 5e-5, one epoch, and a task mix including subtitles. Keeping an arm
    that claims their recipe and carries GRAM's is how the 37.4 became quotable in the first
    place."""
    import os
    assert not os.path.exists('config/sca/ablations/H1_hypergram_paper.json')
    src = open('slurm_scripts/b_grid_pretrain.sh').read()
    assert 'H1_hypergram_paper' not in src.split('ARMS=(')[1].split(')')[0]



def test_gram_lora_is_marked_appendix_not_a_comparison_row():
    """GRAM+LoRA is our construction, not a method anyone proposed. In a comparison table a
    reader would fairly ask who claimed it. It stays as the control that separates geometry
    from adapter -- SCA changes both at once -- and that argument belongs with the ablations."""
    src = open('slurm_scripts/repro_baselines_eval.sh').read()
    assert 'APPENDIX row' in src and 'gram_lora' in src.split('APPENDIX row')[1][:200]
    assert 'MAIN TABLE rows' in src
    main = src.split('MAIN TABLE rows')[1].split('APPENDIX row')[0]
    assert 'gram_lora' not in main, 'gram_lora is listed among the main-table rows'
    for m in ('pmrl', 'hypergram', 'sca'):
        assert m in main, '%s missing from the main-table rows' % m


def test_the_arm_is_part_of_the_cell_identity():
    """Two checkpoints of one method share a config file byte for byte -- gram_hyp2 at lr 2e-5
    and h1_hypergram_paper at 1e-4 differ only in which arm is passed. With the arm absent from
    the output path they collide, and because the fingerprint is computed from the config the
    second run is silently SKIPPED: the first checkpoint's numbers keep the method's name.

    That was harmless while cell_is_done was broken under Slurm. It is not harmless now."""
    src = open('slurm_scripts/repro_baselines_eval.sh').read()
    assert 'OUT="workdir/e1_repro/${METHOD}_${ARM}_${BENCH}"' in src, \
        'the output path must distinguish two checkpoints of the same method'


def test_cell_names_with_an_arm_still_parse():
    import importlib.util
    sp = importlib.util.spec_from_file_location('rvi_parse', 'scripts/raw_vs_itm.py')
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    for cell, bench in (('hypergram_gram_hyp2_msrvtt', 'msrvtt'),
                        ('hypergram_h1_hypergram_paper_didemo', 'didemo'),
                        ('pmrl_pmrl_lora_activitynet', 'activitynet')):
        arm, got = m.split_cell(cell)
        assert got == bench, '%s parsed as benchmark %r' % (cell, got)
        assert arm and arm != cell, '%s did not split into arm and benchmark' % cell


def _resolved(path):
    c = json.load(open(path))['model_cfg']
    d = c.get('default')
    base = json.load(open(d)) if d else {}   # the default file is FLAT, not wrapped
    out = dict(base)
    out.update(c)
    return out


@pytest.mark.parametrize('method,mode', [('pmrl', 'pmrl_raw'), ('hypergram', 'volume'),
                                         ('gram_lora', 'volume')])
@pytest.mark.parametrize('bench', BENCHES)
def test_score_mode_is_stated_not_inherited(method, mode, bench):
    """configs_qweight inherits config/sca/default_model_cfg.json, which sets
    score_mode=centroid. A baseline that merely omitted the key therefore read as None in the
    file and ran as CENTROID -- every competing aggregator scored with ours, and the launcher's
    file-level check said the config matched. score_mode must be explicit per method."""
    p = 'benchmark_eval/configs_repro/%s_%s.json' % (method, bench)
    assert json.load(open(p))['model_cfg'].get('score_mode') == mode, 'not stated in the file'
    assert _resolved(p)['score_mode'] == mode, 'resolves to something else at run time'


def test_no_baseline_resolves_to_the_centroid():
    for p in glob.glob('benchmark_eval/configs_repro/*.json'):
        assert _resolved(p)['score_mode'] != 'centroid', '%s scores a baseline with ours' % p


def test_the_launcher_compares_the_resolved_config():
    src = open('slurm_scripts/repro_baselines_eval.sh').read()
    assert 'def resolved(' in src, 'the check still reads the config file only'
    assert 'RESOLVED config matches' in src
