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
