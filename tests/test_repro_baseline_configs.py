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
    expect = {'pmrl': ('pmrl', 'pmrl_raw'),
              'hypergram': ('gram_hyp', None),
              'gram_lora': ('gram_lora', None)}[method]
    assert m.get('model_type') == expect[0], m.get('model_type')
    assert m.get('score_mode') == expect[1], m.get('score_mode')


def test_the_launcher_refuses_a_config_that_does_not_match_the_checkpoint():
    """Scoring a PMRL checkpoint through the volume, or a hypergraph checkpoint through the
    centroid, yields a full set of plausible numbers for a model that was never trained: no
    shape mismatch, no error, and a table wrong in a way nobody can see."""
    src = open('slurm_scripts/repro_baselines_eval.sh').read()
    assert 'was trained as' in src, 'no model_type cross-check against the arm hps.json'
    assert "score_mode=%r, the arm trained with" in src, 'no score_mode cross-check'


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


def test_the_status_file_exists_and_states_the_recipe_defect():
    """A record that keeps being re-litigated from stale numbers needs one place to live."""
    txt = open('experiments/results/HYPERGRAM_STATUS.md').read()
    assert 'NEVER TRAINED' in txt, 'the status of the corrected arm must be explicit'
    assert '2e-5' in txt and '1e-4' in txt, 'the recipe difference is the whole point'
    assert 'gram_hyp_paper' in txt


def test_the_corrected_arm_differs_from_the_collapsed_one_only_in_the_recipe():
    a = json.load(open('config/baselines/pretrain_cfg/gram_hyp2_pretrain.json'))
    b = json.load(open('config/sca/ablations/H1_hypergram_paper.json'))
    assert a['model_cfg'] == b['model_cfg'], \
        'H1 must be the SAME hyperbolic reading as gram_hyp2 -- only the recipe may differ'
    assert b['run_cfg']['learning_rate'] == 1e-4, b['run_cfg']['learning_rate']
    assert b['data_cfg']['train'][0]['batch_size'] == 128
    assert b['run_cfg']['valid_freq'] == 10, 'a collapse must be locatable, not inferred'


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
