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
import re


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
    assert 'NOT an equal-data comparison' in SRC
    assert 'FATAL: they train on' in SRC


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


def test_the_missing_vendored_package_is_supplied_by_symlink_not_by_editing():
    """Their repo ships no evaluation_tools/ -- the caption-eval package both forks inherit
    from VAST -- and evaluation/evaluation_mm.py imports it at module level, so their code
    does not import at all without it. That is a packaging omission, not a difference in
    method, and the fix must be a symlink rather than a patch: copying files in, or editing
    their imports, would make the run our code under their name."""
    assert 'ln -s "$CODE_DIR/evaluation_tools"' in LAUNCH
    assert 'packaging omission on' in LAUNCH
    # and the dirty check must not then reject the very link it created
    assert "':!evaluation_tools'" in LAUNCH, \
        'the dirty check would refuse the checkout it just linked into'


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
        'their root must be on PYTHONPATH, since sys.path[0] under torchrun is the ranks\\' cwd'
    # and it must come FIRST, or our modules would shadow theirs
    line = [l for l in LAUNCH.splitlines() if l.startswith('export PYTHONPATH=')][0]
    assert line.index('$HG_ROOT') < line.index('PYTHONPATH:+'), \
        'HG_ROOT must precede the inherited path so their code still wins'


def test_a_failed_link_is_fatal_rather_than_silent():
    """`ln -s ... && echo` swallowed the failure: no link, no message, and the job ran on to
    die inside torchrun instead."""
    assert 'could not link evaluation_tools' in LAUNCH
    assert 'ln -s "$CODE_DIR/evaluation_tools" "$HG_ROOT/evaluation_tools" || {' in LAUNCH
