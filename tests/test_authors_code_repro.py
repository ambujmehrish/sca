"""Baselines run from the authors' code, not our reimplementation of it.

Our gram_hyp differs from github.com/uta-smile/HyperGram in six substantive ways -- no
learnable curvature, no curvature learning-rate group, and no scale matching between the
Euclidean and hyperbolic volumes before mixing. So the reproduction runs their repository, and
the only edits are dataset and checkpoint paths.

The property worth protecting is narrow: a path rewrite must not move a hyperparameter, and
their checkout must not be edited. A number labelled "authors' code" that came from a patched
tree or a changed learning rate is worse than no number, because it looks verifiable.
"""
import json
import os
import pathlib
import re

import pytest


SRC = open('scripts/make_hypergram_config.py').read()
LAUNCH = open('slurm_scripts/hypergram_authors.sh').read()


def test_the_rewrite_asserts_hyperparameters_survived():
    """Intent is not enough -- the script checks after editing, and exits on any drift."""
    assert 'Only paths may be rewritten' in SRC
    for key in ('learning_rate', 'batch_size', 'task', 'curvature_init', 'learn_curvature'):
        assert key in SRC, 'the frozen-key check does not cover %s' % key


def test_geometry_mode_is_the_only_hyperparameter_the_script_may_set():
    """It names which of their methods runs, so it has to be settable -- and it is explicitly
    excluded from the frozen list rather than silently absent from it."""
    assert 'FROZEN_MODEL' in SRC
    frozen = re.search(r'FROZEN_MODEL = \((.*?)\)', SRC, re.S).group(1)
    assert 'geometry_mode' not in frozen, 'geometry_mode must not be in the frozen set'
    assert "the ONE hyperparameter this script is allowed to set" in SRC


def test_an_annotation_substitution_is_refused_by_default():
    """They train on annotations150k_clean.json and we have annotations150k.json. Swapping one
    for the other without saying so makes the comparison unequal in the training data."""
    assert '--allow_annotation_mismatch' in SRC
    assert 'FATAL: they train on' in SRC
    # and the substitute must be checked against OUR reported row, not merely allowed:
    # a third training set is unequal to their data AND to ours
    assert '--sca_train_config' in SRC
    assert 'is NOT the file our SCA row trains on' in SRC
    assert 'never as reproducing' in SRC, \
        'the recorded note must forbid citing the row as their published number'


def test_the_launcher_refuses_a_modified_checkout():
    """A local edit makes the run our code under their name."""
    assert 'has local modifications' in LAUNCH
    assert 'OUR code under THEIR name' in LAUNCH
    assert 'git -C "$HG_ROOT" rev-parse --short HEAD' in LAUNCH, \
        'the commit must be logged, so a row can be traced to a revision'


def test_the_pmrl_recipe_caveat_is_carried_into_the_generated_config():
    """No PMRL config ships with their repo, so pmrl* modes inherit HyperGRAM's recipe. That
    is their implementation at another paper's hyperparameters, and the distinction has to
    survive into whatever reads the config later."""
    assert 'recipe_caveat' in SRC
    assert "never as PMRL's published setup" in SRC
    assert 'RECIPE CAVEAT' in LAUNCH


def test_all_four_geometries_run_from_one_codebase():
    modes = re.search(r'MODES=\((.*?)\)', LAUNCH).group(1).split()
    assert modes == ['hybrid', 'pmrl', 'pmrl_volume', 'hybrid_pmrl'], modes
    assert '--array=0-3' in LAUNCH


def test_the_superseded_reimplementation_rows_are_refused():
    """repro_baselines_eval.sh evaluated OUR pmrl and hypergram. Both are now known not to
    match the released code, so those indices must not run at all -- a retired baseline that
    still answers to `sbatch` is how it reappears in a table months later. gram_lora stays
    reachable: it is ours by construction and is only ever an appendix control."""
    import subprocess
    r = subprocess.run(['bash', 'slurm_scripts/repro_baselines_eval.sh'],
                       env={'SLURM_ARRAY_TASK_ID': '0', 'PATH': '/usr/bin:/bin'},
                       capture_output=True, text=True)
    assert r.returncode == 2, r.stdout + r.stderr
    assert 'RETIRED' in r.stderr
    assert 'hypergram_authors.sh' in r.stderr, 'the refusal must name the replacement'

    src = open('slurm_scripts/repro_baselines_eval.sh').read()
    guard = src.index('are RETIRED')
    env = src.index('sca_env.rc')
    assert guard < env, 'the guard must fire before the environment is sourced, or a missing '\
                        'env masks it'
    assert 'RETIRED -- do not run' in src.split('\n')[15:25][0] or 'RETIRED' in src[:2000]


def test_the_missing_directories_are_supplied_by_symlink_not_by_editing():
    """Their repo ships neither evaluation_tools/ -- the caption-eval package both forks
    inherit from VAST, imported at module level by evaluation/evaluation_mm.py -- nor
    pretrained_weights/, the encoder checkpoints their own default_model_cfg.json names and
    their model code loads by relative path. Both are packaging omissions, not differences in
    method, and the fix must be a symlink rather than a patch: copying files in, or editing
    their paths, would make the run our code under their name."""
    assert 'ln -s "$src" "$dst"' in LAUNCH, 'the dependency must be linked, not copied'
    for dep in ('evaluation_tools', 'pretrained_weights'):
        assert 'link_dep %s ' % dep in LAUNCH, 'no dependency link for %s' % dep
        # and the dirty check must not then reject the very link it created
        assert "':!%s'" % dep in LAUNCH, \
            'the dirty check would refuse the %s link it just created' % dep


def test_val_annotations_come_from_our_tree_with_a_loud_failure_if_absent():
    """Their val block names datasets/annotations/<bench>/... and they ship no datasets/
    directory, so those paths resolve nowhere in their tree. Ours supplies them -- the same
    VAST-family files -- which also keeps the eval split identical to every other row in our
    table. If one is missing the generator must say so rather than hand their code a path that
    does not exist."""
    assert 'their repo ships no' in SRC
    assert 'val annotation' in SRC and 'FATAL' in SRC


def test_the_dependency_link_is_verified_by_importing_not_by_existing():
    """The first attempt created the symlink and still died on ModuleNotFoundError across all
    four ranks. Existence on disk is not the property that matters -- importability from the
    directory their ranks actually start in is, and srun does not necessarily inherit the
    subshell's cwd. So the launcher imports it for real before spending a node, and names
    their root on PYTHONPATH so the import does not depend on cwd at all."""
    assert 'python3 -c "import evaluation_tools"' in LAUNCH, \
        'the link is assumed to work rather than checked'
    assert 'does not import from there' in LAUNCH
    assert 'export PYTHONPATH="$HG_ROOT' in LAUNCH, \
        "their root must be on PYTHONPATH: sys.path[0] under torchrun is the ranks' cwd"
    # and it must come FIRST, or our modules would shadow theirs
    line = [l for l in LAUNCH.splitlines() if l.startswith('export PYTHONPATH=')][0]
    assert line.index('$HG_ROOT') < line.index('PYTHONPATH:+'), \
        'HG_ROOT must precede the inherited path so their code still wins'


def test_a_failed_link_is_fatal_rather_than_silent():
    """`ln -s ... && echo` swallowed the failure: no link, no message, and the job ran on to
    die inside torchrun instead."""
    assert 'could not link $name' in LAUNCH
    assert 'ln -s "$src" "$dst" || {' in LAUNCH
    for dep in ('evaluation_tools', 'pretrained_weights'):
        call = [l for l in LAUNCH.splitlines() if l.startswith('link_dep %s ' % dep)]
        assert call and call[0].endswith('|| exit 2'), \
            'link_dep %s must abort the job when it fails, got %r' % (dep, call)


def _link_dep(tmp_path, name, code_has=True, preexisting=None):
    """Run the launcher's own link_dep in a sandbox. Extracted rather than reimplemented --
    a test of a copy of the function proves nothing about the one that runs on the cluster."""
    import subprocess
    fn = re.search(r'^link_dep\(\) \{.*?^\}', LAUNCH, re.S | re.M).group(0)
    code, hg = tmp_path / 'code', tmp_path / 'hg'
    (code / name).mkdir(parents=True, exist_ok=True) if code_has \
        else code.mkdir(parents=True, exist_ok=True)
    hg.mkdir(exist_ok=True)
    if preexisting is not None:
        (hg / name).unlink(missing_ok=True)
        (hg / name).symlink_to(preexisting)
    r = subprocess.run(['bash', '-c', '%s\nlink_dep %s "a dep"' % (fn, name)],
                       capture_output=True, text=True,
                       env={'CODE_DIR': str(code), 'HG_ROOT': str(hg), 'PATH': '/usr/bin:/bin'})
    return r, hg / name


def test_a_dangling_link_is_replaced_rather_than_reported_as_present(tmp_path):
    """The state that killed job 53767117 four seconds in: `-e` follows the symlink and says
    absent, `ln` looks at the link itself and says "File exists". The old code's else-branch
    printed "already present" for exactly this -- which is very likely what the third
    evaluation_tools attempt did before dying on import."""
    r, dst = _link_dep(tmp_path, 'dep', preexisting='/nonexistent/place')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'DANGLING' in r.stdout and '/nonexistent/place' in r.stdout, \
        'the old target must be logged, not silently discarded'
    assert dst.exists() and dst.resolve() == (tmp_path / 'code' / 'dep')


def test_linking_is_idempotent_and_resolution_is_what_is_checked(tmp_path):
    r, dst = _link_dep(tmp_path, 'dep', preexisting=None)
    assert r.returncode == 0 and dst.exists()
    r2, _ = _link_dep(tmp_path, 'dep', preexisting=str(tmp_path / 'code' / 'dep'))
    assert r2.returncode == 0
    assert 'already present' in r2.stdout and 'code/dep' in r2.stdout, \
        'the existing branch must say where it actually resolves to'


def test_nothing_to_supply_from_is_fatal(tmp_path):
    r, _ = _link_dep(tmp_path, 'dep', code_has=False)
    assert r.returncode == 1
    assert 'nothing to supply from' in r.stderr


def test_the_encoder_weights_are_checked_before_a_node_is_spent():
    """Their model code loads encoder checkpoints by relative path from inside model
    construction, so a missing file surfaces as a torchrun traceback on all four ranks after
    the run has already started -- which is how EVA01_CLIP_g_14 was discovered, 38 seconds
    into a four-way array. The check reads the RESOLVED config rather than assuming the
    encoder, and an unrecognised encoder name is fatal instead of skipped."""
    assert 'vision_encoder_type' in LAUNCH and 'audio_encoder_type' in LAUNCH
    assert 'default_model_cfg.json' in LAUNCH, \
        'the encoder type comes from their defaults merged with this run"s model_cfg'
    assert 'evaclip01_giant' in LAUNCH and 'BEATs_iter3_plus_AS2M.pt' in LAUNCH
    assert 'bert/bert-base-uncased' in LAUNCH
    assert 'is not one this pre-flight knows a weight file' in LAUNCH, \
        'an unknown encoder must be fatal, not silently unchecked'


def test_the_ranks_are_placed_in_their_root_explicitly():
    """Every relative path their code hardcodes resolves inside the RANKS, and the subshell's
    `cd` only sets the cwd of srun itself."""
    assert 'srun --chdir="$HG_ROOT"' in LAUNCH


HG = '/home/user/uta-smile/hypergram'
# Their config's val block names datasets/annotations/vatex/descs_ret_test.json; our tree has
# the _431 subset (the VATEX videos we actually downloaded). The generator substitutes it under
# the same flag, so the tests need it on disk.
VATEX_SUBSET = pathlib.Path('datasets/annotations/vatex/descs_ret_test_431.json')


@pytest.fixture
def hypergram_sandbox(tmp_path):
    """A data root the generator can succeed against, and the real checkout to read from."""
    if not os.path.isdir(HG + '/configs/pretrain'):
        pytest.skip('no HyperGram checkout at %s' % HG)
    data = tmp_path / 'data' / 'vast27m_150k'
    data.mkdir(parents=True)
    # their frozen task is ret%tvas -- the `s` requires every entry to carry a subtitle
    (data / 'annotations150k.json').write_text(json.dumps(
        [{'clip_id': 'v%d' % i, 'desc': 'a caption', 'subtitle': 'a subtitle'}
         for i in range(4)]))
    (tmp_path / 'vast').mkdir()
    made = not VATEX_SUBSET.exists()
    if made:
        VATEX_SUBSET.write_text('[]')
    generated = []
    yield tmp_path, generated
    for g in generated:
        if os.path.exists(g):
            os.remove(g)
    if made:
        VATEX_SUBSET.unlink()


def _generate(sandbox, extra=()):
    """Run the generator end to end against the real checkout, in the sandboxed data root."""
    import subprocess
    tmp_path, generated = sandbox
    out = 'configs/pretrain/repro_pytest_%s.json' % tmp_path.name
    generated.append(HG + '/' + out)
    r = subprocess.run(
        ['python3', 'scripts/make_hypergram_config.py', '--hypergram_root', HG,
         '--data_root', str(tmp_path / 'data'), '--vast_ckpt_dir', str(tmp_path / 'vast'),
         '--out', out] + list(extra), capture_output=True, text=True)
    gen = HG + '/' + out
    return r, (json.load(open(gen)) if os.path.exists(gen) else None)


def test_the_substituted_training_file_is_actually_written_into_the_config(hypergram_sandbox):
    """The note once said SUBSTITUTED while the config still named THEIR path -- the run
    would have gone looking for a file that does not exist, with a config claiming otherwise.
    What the note says and what the config says have to be the same thing."""
    r, cfg = _generate(hypergram_sandbox, ['--allow_annotation_mismatch'])
    assert r.returncode == 0, r.stdout + r.stderr
    txt = cfg['data_cfg']['train'][0]['txt']
    expect = str(hypergram_sandbox[0] / 'data' / 'vast27m_150k' / 'annotations150k.json')
    assert txt == expect, txt
    assert 'SUBSTITUTED' in cfg['_repro_note']['annotations']


def test_every_data_path_in_the_generated_config_exists(hypergram_sandbox):
    """A generated config is only useful if their code can actually open what it names."""
    r, cfg = _generate(hypergram_sandbox, ['--allow_annotation_mismatch'])
    assert r.returncode == 0, r.stdout + r.stderr
    checked = 0
    for blk in list(cfg['data_cfg']['train']) + list(cfg['data_cfg'].get('val', [])):
        for key in ('txt',):
            assert os.path.exists(blk[key]), '%s %s -> %s does not exist' % (
                blk.get('name', 'train'), key, blk[key])
            checked += 1
    assert checked >= 2, 'nothing was actually checked'
    assert os.path.isdir(cfg['run_cfg']['pretrain_dir'])


def test_the_substitution_is_refused_without_the_flag(hypergram_sandbox):
    r, cfg = _generate(hypergram_sandbox)
    assert r.returncode == 1 and cfg is None
    assert 'they train on annotations150k_clean.json' in r.stdout + r.stderr


def test_hyperparameters_survive_the_generation(hypergram_sandbox):
    r, cfg = _generate(hypergram_sandbox, ['--allow_annotation_mismatch'])
    assert r.returncode == 0
    assert cfg['run_cfg']['learning_rate'] == 5e-05
    assert cfg['data_cfg']['train'][0]['epoch'] == 1
    assert cfg['data_cfg']['train'][0]['task'] == 'ret%tvas%tv%ta'
    assert cfg['data_cfg']['train'][0]['batch_size'] == 128
    assert cfg['model_cfg']['learn_curvature'] is True


def _rewrite_annos(sandbox, entries):
    tmp_path, _ = sandbox
    (tmp_path / 'data' / 'vast27m_150k' / 'annotations150k.json').write_text(json.dumps(entries))


def test_a_subtitleless_file_is_refused_because_their_code_would_not_complain(hypergram_sandbox):
    """model/gram.py picks the Gramian arity with `if "raw_subtitles" in batch.keys()`, so an
    annotation file with no subtitle field turns their published 4-modality method into a
    3-modality one and trains to completion. The number would carry their name having never
    run their method -- the failure mode that produces a wrong table rather than a crash."""
    _rewrite_annos(hypergram_sandbox, [{'clip_id': 'v', 'desc': 'c'}] * 4)
    r, cfg = _generate(hypergram_sandbox, ['--allow_annotation_mismatch'])
    assert r.returncode == 1 and cfg is None
    out = r.stdout + r.stderr
    assert 'requires subtitles' in out and 'hybrid_volume3' in out


def test_partial_subtitle_coverage_is_refused(hypergram_sandbox):
    """annoindexedcollate keeps or drops raw_subtitles based on `data[0] is None` -- the FIRST
    sample of the batch. Partial coverage therefore flips the volume rank between batches and
    hands None to the tokenizer for the rest."""
    _rewrite_annos(hypergram_sandbox,
                   [{'clip_id': 'a', 'desc': 'c', 'subtitle': 's'}, {'clip_id': 'b', 'desc': 'c'}])
    r, cfg = _generate(hypergram_sandbox, ['--allow_annotation_mismatch'])
    assert r.returncode == 1 and cfg is None
    assert 'FIRST sample alone' in r.stdout + r.stderr


def test_full_subtitle_coverage_is_recorded(hypergram_sandbox):
    r, cfg = _generate(hypergram_sandbox, ['--allow_annotation_mismatch'])
    assert r.returncode == 0, r.stdout + r.stderr
    assert cfg['_repro_note']['subtitle_coverage'] == 1.0


def test_the_task_parser_reads_every_retrieval_group(hypergram_sandbox):
    import importlib.util
    spec = importlib.util.spec_from_file_location('m', 'scripts/make_hypergram_config.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.task_modalities('ret%tvas%tv%ta') == set('tvas')
    assert m.task_modalities('ret%tv%ta') == set('tva')
    assert 's' not in m.task_modalities('ret%tv%ta'), \
        'our own arms run three modalities; the check must not fire on them'
