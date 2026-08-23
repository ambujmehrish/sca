"""PMRL from its authors' RELEASED checkpoint, not from anyone's reimplementation.

The PMRL authors publish code (github.com/Xiaohao-Liu/PMRL) and weights
(huggingface.co/xhLiu/PMRL). So the row needs no training: it is their checkpoint, our
environment, our evaluation protocol -- the same standing as the GRAM released-checkpoint row.

Two properties are worth protecting. The evaluation protocol must be the one every other row
uses, or this becomes the single differently-measured row. And the checkpoint must be shown to
have LOADED: their build_model uses strict=False, so a mismatch produces a partly random model
that evaluates without error.
"""
import re

SRC = open('scripts/make_pmrl_config.py').read()
LAUNCH = open('slurm_scripts/pmrl_released.sh').read()
COMMON = open('scripts/repro_common.py').read()


def test_nothing_is_retrained():
    assert '--mode' not in LAUNCH or 'testing' in SRC
    assert "cfg['run_cfg']['mode'] = 'testing'" in SRC
    assert "cfg['run_cfg']['zero_shot'] = True" in SRC


def test_the_protocol_is_ours_so_every_row_is_measured_alike():
    """Their zero-shot config evaluates VATEX at 16 frames on the full test list; ours uses 8
    and the 431 subset. Using theirs would make PMRL the one row measured differently."""
    assert 'benchmark_eval/configs_e1/gram_%s.json' in SRC
    assert 'single-configuration rule' in SRC
    assert 'evaluation_protocol' in SRC


def test_model_type_is_set_because_the_default_would_fail_silently():
    """Their default_model_cfg says model_type "vast" and the released checkpoint is PMRL.
    build_model loads strict=False, so a VAST skeleton would absorb what it recognised,
    randomly initialise the rest, and report numbers."""
    assert "cfg['model_cfg']['model_type'] = 'pmrl'" in SRC
    assert 'strict=False' in SRC


def test_frozen_model_keys_are_checked_after_the_rewrite():
    frozen = re.search(r'FROZEN_MODEL = \((.*?)\)', SRC, re.S).group(1)
    for k in ('itm_rerank_num', 'evaluation_type', 'vision_encoder_type'):
        assert k in frozen, 'the frozen set does not cover %s' % k
    assert 'Only model_type may be set' in SRC
    assert 'model_type' not in frozen


def test_the_checkpoint_load_is_verified_not_assumed():
    """The whole row rests on the released weights actually being in the model."""
    assert 'missing_keys' in LAUNCH and 'Unexpected keys' in LAUNCH
    assert 'must not be reported' in LAUNCH
    assert 'does not record missing/unexpected keys' in LAUNCH, \
        'a log that never printed the keys must fail, not pass by default'


def test_the_missing_vast_module_is_a_stub_that_refuses_to_run():
    """model/__init__.py does `from .vast import VAST` and model/vast.py is not shipped, so
    `import model` raises before anything runs. PMRL never constructs VAST, so a stub is
    enough -- but a stub that returned a usable object would fabricate a baseline, and
    strict=False means the fabrication would evaluate silently."""
    assert 'from .vast import VAST' in LAUNCH
    assert 'raise NotImplementedError' in LAUNCH
    assert 'model/vast.py' in LAUNCH
    stub = LAUNCH[LAUNCH.index("cat > \"$STUB\""):LAUNCH.index('PYEOF\n  echo "wrote $STUB')]
    assert 'return' not in stub.split('class VAST')[1], 'the stub must raise, never return'


def test_supplied_dependencies_are_excluded_from_the_dirty_check():
    for dep in ('config/vast', 'pretrained_weights', 'datasets', 'model/vast.py'):
        assert "':!%s'" % dep in LAUNCH, 'the dirty check would reject %s' % dep
    assert 'OUR code under THEIR name' in LAUNCH


def test_the_audio_filter_helper_is_shared_between_both_reproductions():
    """Both released repos carry the same hardcoded `<id>.mp3` filter. A helper copy-pasted
    into the second generator is how the first one silently stops being checked."""
    assert 'def resolve_audio_dir' in COMMON
    assert 'from repro_common import' in SRC
    assert 'from repro_common import' in open('scripts/make_hypergram_config.py').read()
    assert 'def resolve_audio_dir' not in SRC
