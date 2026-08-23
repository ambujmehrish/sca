#!/usr/bin/env python3
"""Run another project's entry point with a non-fork multiprocessing start method.

    python3 -m torch.distributed.launch ... run_with_forkserver.py ./run.py --config ...

WHY. Their `utils/build_dataloader.py` constructs `DataLoader(..., num_workers=8)` without
passing `multiprocessing_context`, so workers are forked. Something in this environment
refuses to operate in a forked child:

    This operation is not valid in a forked process. Original PID=..., current PID=...
    Avoid using os.fork() and make sure your multiprocessing start method is
    'forkserver' or 'spawn'.

In a DataLoader worker that exception lands inside `AudioMapper.read`, whose `except
Exception` prints it and returns None; `IndexAnno.__getitem__` then raises a bare ValueError
because `self.training` is False. The visible failure is thirteen frames away from the cause.

Our own fork sets the context from $GRAM_MP_CTX for exactly this reason. Theirs has no such
option, and editing their code would make the result ours under their name -- so the start
method is set here, in the process that launches theirs, before torch or CUDA is touched.

This changes no behaviour of their model: it changes how worker processes are created.
"""
import multiprocessing
import os
import runpy
import sys


def find_script(argv):
    """The target script, wherever it sits in argv.

    `torch.distributed.launch` without --use-env PREPENDS `--local-rank=N` to the script's
    arguments, so the script is NOT argv[1]:

        run_with_forkserver.py --local-rank=0 ./run.py --config ...

    Their run.py reads that rank through utils/args.py, so it has to be passed along rather
    than swallowed. Returns (script, args_for_it) with the rank preserved in order.
    """
    for i, a in enumerate(argv):
        if a.endswith('.py') and os.path.exists(a):
            return a, argv[:i] + argv[i + 1:]
    return None, argv


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: run_with_forkserver.py [--local-rank=N] <script.py> [args...]')

    method = os.environ.get('GRAM_MP_CTX', 'forkserver')
    if method not in multiprocessing.get_all_start_methods():
        sys.exit('FATAL: multiprocessing start method %r is unavailable here (have: %s).'
                 % (method, ', '.join(multiprocessing.get_all_start_methods())))
    # force=True because torch may already have set 'fork' as a side effect of being imported
    # by something earlier in the launcher chain.
    multiprocessing.set_start_method(method, force=True)
    got = multiprocessing.get_start_method()
    if got != method:
        sys.exit('FATAL: asked for start method %r, got %r.' % (method, got))

    script, rest = find_script(sys.argv[1:])
    if script is None:
        sys.exit('FATAL: no existing .py file among %r (cwd is %s). torch.distributed.launch\n'
                 '       prepends --local-rank=N, so the script is not necessarily the first\n'
                 '       argument.' % (sys.argv[1:], os.getcwd()))
    print('[run_with_forkserver] start method %s, running %s %s'
          % (got, script, ' '.join(rest)), flush=True)

    # Their run.py reads sys.argv through utils/args.py and expects argv[0] to be itself.
    # --local-rank stays in the arguments: their code needs it.
    sys.argv = [script] + rest
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
