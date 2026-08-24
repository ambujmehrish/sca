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
    """(missing, unexpected) or (None, None) when the log never printed them."""
    def grab(label):
        m = re.search(re.escape(label) + r'\s*\[(.*?)\]', text, re.S)
        if not m:
            return None
        body = m.group(1).strip()
        return [k.strip().strip("'\"") for k in body.split(',') if k.strip()] if body else []
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
