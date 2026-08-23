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
    stub = LAUNCH[LAUNCH.index('cat > "$STUB_TMP"'):LAUNCH.index('PYEOF\n  mv -f "$STUB_TMP"')]
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


def _link_dep_fn():
    import re as _re
    return _re.search(r'^link_dep\(\) \{.*?^\}', LAUNCH, _re.S | _re.M).group(0)


def test_concurrent_array_tasks_do_not_corrupt_the_dependency_links(tmp_path):
    """Five array tasks share one checkout. With a test-then-create `ln -s`, all five saw
    config/vast absent, all five created it, and the losers found it already resolving to a
    DIRECTORY -- so ln placed the link INSIDE it, leaving the stray config/pmrl/pmrl that
    then tripped the dirty-checkout guard. `ln -sfn` converges instead."""
    import subprocess
    pm, src = tmp_path / 'pm', tmp_path / 'src'
    (pm / 'config' / 'pmrl').mkdir(parents=True)
    src.mkdir()
    script = '%s\nlink_dep config/vast "%s" t\n' % (_link_dep_fn(), pm / 'config' / 'pmrl')
    procs = [subprocess.Popen(['bash', '-c', script], env={'PM_ROOT': str(pm),
                                                           'PATH': '/usr/bin:/bin'},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for _ in range(5)]
    for p in procs:
        p.wait()
    assert (pm / 'config' / 'vast').is_symlink()
    assert not (pm / 'config' / 'pmrl' / 'pmrl').exists(), \
        'a racing task descended into the symlink instead of replacing it'


def test_a_real_directory_is_never_clobbered_by_the_link(tmp_path):
    import subprocess
    pm, src = tmp_path / 'pm', tmp_path / 'src'
    (pm / 'realdir').mkdir(parents=True)
    (pm / 'realdir' / 'keepme').write_text('x')
    src.mkdir()
    r = subprocess.run(['bash', '-c', '%s\nlink_dep realdir "%s" t\n' % (_link_dep_fn(), src)],
                       env={'PM_ROOT': str(pm), 'PATH': '/usr/bin:/bin'},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (pm / 'realdir' / 'keepme').exists(), 'a real directory was replaced by a symlink'


def test_bytecode_caches_do_not_count_as_a_modified_checkout():
    """__pycache__ appears because we IMPORT their modules to verify the package loads. That
    is a byproduct of running their code, not an edit to it, and treating it as a local
    modification made the guard refuse a clean checkout."""
    assert "':!*__pycache__*'" in LAUNCH


def test_the_import_stub_is_installed_atomically():
    """Five tasks writing the same file directly can interleave into a truncated module."""
    assert 'STUB_TMP' in LAUNCH and 'mv -f "$STUB_TMP" "$STUB"' in LAUNCH


def test_log_name_is_supplied_because_their_default_omits_it():
    """run.py:28 reads run_cfg.log_name for wandb.init and their released default_run_cfg.json
    does not define it, so the run died there having already loaded a 5.6 GB checkpoint. It is
    a logging label; supplying it changes no measured quantity, and it is recorded as an
    addition rather than passed off as theirs."""
    assert "cfg['run_cfg']['log_name']" in SRC
    assert 'run.py:28' in SRC
    assert "'log_name':" in SRC, 'the addition must be recorded in _repro_note'


def test_the_load_check_is_skipped_when_the_run_itself_failed():
    """Otherwise a crash is reported as "the log does not record missing/unexpected keys",
    which describes a different problem and hides the real one."""
    assert 'if [ $rc -ne 0 ]; then' in LAUNCH
    assert 'The real error is above this line' in LAUNCH
    # anchored to the invocation, not to prose: the message is also quoted in the comment
    i_guard = LAUNCH.index('if [ $rc -ne 0 ]; then')
    i_check = LAUNCH.index('python3 - "$OUT/run.log"')
    assert i_guard < i_check, 'the skip must come before the check it guards'
