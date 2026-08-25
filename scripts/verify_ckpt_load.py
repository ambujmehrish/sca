#!/usr/bin/env python3
"""Did a checkpoint ACTUALLY load into the model that was scored?

    python3 scripts/verify_ckpt_load.py <eval log file> [--allow-prefix lora_]

Our trunk, like every VAST fork, loads checkpoints with strict=False and only LOGS the
mismatch (missing_keys / unexpected keys). A checkpoint that does not fit therefore produces
numbers from a partly random model without any error -- the exact failure the PMRL row's
verifier caught on the authors' side. This is the same check for OUR eval logs, used
wherever a released checkpoint is scored through our classes (the masked-eval sweep): a
non-empty list, or a log that never printed the lists, fails loudly.
"""
import argparse
import re
import sys


def keylists(text):
    """(missing, unexpected) or (None, None) when the log never printed them.

    Parsed ONE LINE AT A TIME, over EVERY occurrence. Four ranks each print their own
    missing_keys/Unexpected keys line into the same tee'd log, so a bracket opened by one
    rank can be closed after another rank's line has interleaved. The previous regex was
    `(.*?)` under re.S on the whole file and took the FIRST match, so it could swallow
    interleaved text and report it as key names -- nondeterministically, since interleaving
    is a race. That is exactly what happened: the same released checkpoint verified clean on
    23 of 25 sweep cells and was 'refused' on the other two. A verifier whose verdict depends
    on process scheduling is worse than none, because it launders a race into a provenance
    claim.

    Line-scoped ([^]\n]*) cannot cross a rank boundary, and the union over occurrences means
    one rank's genuine mismatch is never hidden by another rank's clean line."""
    def grab(label):
        hits = re.findall(re.escape(label) + r'\s*\[([^\]\n]*)\]', text)
        if not hits:
            return None
        keys = []
        for body in hits:
            keys += [k.strip().strip("'\"") for k in body.split(',') if k.strip()]
        return sorted(set(keys))
    return grab('missing_keys'), grab('Unexpected keys')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--allow-prefix', action='append', default=[],
                    help='key prefixes that may be MISSING (e.g. mask-token buffers a '
                         'full-FT checkpoint legitimately lacks); never applies to '
                         'unexpected keys')
    args = ap.parse_args()

    missing, unexpected = keylists(open(args.log, errors='replace').read())
    if missing is None or unexpected is None:
        sys.exit('FATAL: %s never printed missing_keys/Unexpected keys, so it cannot be '
                 'shown that the checkpoint loaded. strict=False means an unverified load '
                 'is a number from a partly random model.' % args.log)
    blocked = [k for k in missing if not any(k.startswith(p) for p in args.allow_prefix)]
    print('checkpoint load: %d missing (%d allowed by prefix), %d unexpected'
          % (len(missing), len(missing) - len(blocked), len(unexpected)))
    if blocked or unexpected:
        for k in blocked[:10]:
            print('  MISSING    %s' % k)
        for k in unexpected[:10]:
            print('  UNEXPECTED %s' % k)
        sys.exit('FATAL: the checkpoint did not fully load into this model class. The cell '
                 'must not be reported.')
    print('loaded completely.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
