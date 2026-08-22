"""An arm is scored with the geometry it was TRAINED with, read from its own config.

frameset_eval.sh used to pick the eval config by matching '*qweight*' in the arm's directory
name. That held while the only query-weighted arm was called t9_qweight_only, and broke
silently the moment the sweep arms existed: g1_r16_qw, s1_t9_seed51 and x3_xenc_clean_lr2e5
are all query-weighted without frame slots, none of them matches the pattern, and all three
would have been routed to configs_frames -- where the frame-slots guard raises because no
per-frame features arrive.

That is the good case. The bad one is configs_e1, which scores a query-weighted checkpoint
with a UNIFORM centroid: no guard fires, every shape is right, and the number reported is for
a model that was never trained. So the routing is derived from model_cfg, and this test pins
the mapping against every arm config on disk.
"""
import glob
import json
import os


def route(model_cfg):
    """The rule frameset_eval.sh implements, in one place so it can be tested."""
    if model_cfg.get('sca_frame_slots'):
        return 'configs_frames'
    if model_cfg.get('sca_query_weighting'):
        return 'configs_qweight'
    return 'configs_e1'


def test_the_launcher_derives_the_config_from_hps_not_from_the_name():
    src = open('slurm_scripts/frameset_eval.sh').read()
    assert "hps.json" in src, 'the launcher no longer reads the arm\'s training config'
    assert "case \"$arm\" in\n      *qweight*" not in src, \
        'name matching is back -- it silently mis-scores every arm not called *qweight*'
    # and it must refuse rather than fall through to a default when hps.json is unreadable
    assert 'refusing to guess the scoring geometry' in src


def test_every_arm_config_routes_somewhere_and_the_directory_exists():
    cfgs = sorted(glob.glob('config/sca/ablations/*.json'))
    assert cfgs, 'no ablation configs found'
    for p in cfgs:
        mc = json.load(open(p))['model_cfg']
        d = os.path.join('benchmark_eval', route(mc))
        assert os.path.isdir(d), '%s routes to %s, which does not exist' % (p, d)
        for bench in ('msrvtt', 'didemo', 'activitynet', 'vatex', 'audiocaps'):
            f = os.path.join(d, 'sca_%s.json' % bench)
            assert os.path.exists(f), '%s routes to %s, missing %s' % (p, d, f)


def test_the_query_weighted_sweep_arms_do_not_route_to_frames():
    """The concrete regression: these are the arms whose names lack 'qweight'."""
    for name in ('G1_r16_qw', 'G2b_r32_a16_qw', 'S1_t9_seed51', 'X3_xenc_clean_lr2e5'):
        mc = json.load(open('config/sca/ablations/%s.json' % name))['model_cfg']
        assert route(mc) == 'configs_qweight', \
            '%s would be scored with the wrong geometry' % name
        assert 'qweight' not in name.lower(), \
            '%s now contains "qweight", so this test no longer covers the failure' % name


def test_a_frame_slot_arm_still_routes_to_frames():
    mc = json.load(open('config/sca/ablations/T6_frameset.json'))['model_cfg']
    assert mc.get('sca_frame_slots') is True
    assert route(mc) == 'configs_frames'
