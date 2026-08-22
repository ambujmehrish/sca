"""Every launcher must be able to find scripts/cell_done.sh when Slurm runs it.

Under sbatch, $0 is a COPY of the batch script at /var/spool/slurmd/job<N>/slurm_script, so
"$(dirname "$0")/../scripts/cell_done.sh" resolves to /var/spool/slurmd/scripts/cell_done.sh,
which does not exist. Every launcher used that form, and the source has been failing in every
Slurm job since it was written.

It never surfaced because of how the result is used: `cell_is_done "$out" "$cfg" && continue`.
An undefined function returns 127, the && short-circuits, and the cell just runs. So the
resume-skip and the config fingerprinting -- the latter added specifically after an eval was
silently skipped under a changed config -- have both been inert under Slurm the whole time,
which is also why eval cells appear twice in the logs. It only became visible when
b_grid_pretrain.sh used `claim_outdir "$OUT" || exit 2`, where 127 kills the job.

A protection that fails open is worse than no protection, because it is reported as working.
"""
import glob
import os
import re
import subprocess

LAUNCHERS = sorted(glob.glob('slurm_scripts/*.sh'))


def _sourcing(path):
    return [l for l in open(path).read().splitlines()
            if 'cell_done.sh' in l and l.strip().startswith(('source', 'HELPER=', '[ -f'))]


def test_no_launcher_resolves_the_helper_through_dollar_zero_alone():
    bad = []
    for p in LAUNCHERS:
        s = open(p).read()
        if 'cell_done.sh' not in s:
            continue
        if 'HELPER=' not in s and '$(dirname "$0")' in s:
            bad.append(p)
    assert not bad, ('these resolve the helper only through $0, which is a spool copy under '
                     'sbatch: %s' % ', '.join(bad))


def test_every_launcher_that_uses_the_helper_aborts_when_it_cannot_be_sourced():
    for p in LAUNCHERS:
        s = open(p).read()
        if 'cell_done.sh' not in s:
            continue
        assert 'FATAL: cannot source' in s, '%s sources the helper without checking' % p
        assert re.search(r'command -v (cell_is_done|claim_outdir) >/dev/null', s), \
            '%s does not verify the function is actually defined afterwards' % p


def test_the_helper_resolves_when_dollar_zero_points_at_a_spool_copy(tmp_path):
    """The real failure, reproduced: run the resolution logic with $0 in a directory that has
    no scripts/ sibling, and check it still finds the helper via CODE_DIR."""
    spool = tmp_path / 'job99'
    spool.mkdir()
    fake = spool / 'slurm_script'
    fake.write_text('#!/bin/bash\n')
    code_dir = os.path.abspath('.')
    script = '''
    set -uo pipefail
    cd "$CODE_DIR"
    HELPER="${CODE_DIR:-.}/scripts/cell_done.sh"
    [ -f "$HELPER" ] || HELPER="$(dirname "%s")/../scripts/cell_done.sh"
    source "$HELPER" || exit 2
    command -v cell_is_done >/dev/null || exit 3
    command -v claim_outdir >/dev/null || exit 4
    echo OK
    ''' % fake
    r = subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                       env={**os.environ, 'CODE_DIR': code_dir})
    assert r.returncode == 0, 'rc=%d stderr=%s' % (r.returncode, r.stderr)
    assert 'OK' in r.stdout


def test_the_old_form_really_does_fail_so_this_is_not_a_vacuous_check(tmp_path):
    """Guards the guard: if $0-relative resolution happened to work, every test above would
    pass for the wrong reason."""
    spool = tmp_path / 'job99'
    spool.mkdir()
    fake = spool / 'slurm_script'
    fake.write_text('#!/bin/bash\n')
    r = subprocess.run(
        ['bash', '-c', 'source "$(dirname "%s")/../scripts/cell_done.sh" 2>/dev/null; '
                       'command -v cell_is_done >/dev/null && echo DEFINED' % fake],
        capture_output=True, text=True)
    assert 'DEFINED' not in r.stdout, \
        'the $0-relative form resolved, so the failure this file is about did not occur'
