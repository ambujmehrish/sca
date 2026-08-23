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
    """Our HyperGRAM reimplementation reaches 37.4 against their published 56.6. That row is
    our reimplementation, never HyperGRAM's performance, and the launcher has to say so where
    whoever runs it will read it."""
    src = open('slurm_scripts/repro_baselines_eval.sh').read()
    assert 'never as HyperGRAM' in src
    assert '37.4' in src and '56.6' in src, 'the size of the reproduction gap must be stated'
