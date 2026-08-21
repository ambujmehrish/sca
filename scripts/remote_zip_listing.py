#!/usr/bin/env python3
"""List a remote zip's contents via HTTP range requests, without downloading it.

    python3 scripts/remote_zip_listing.py --repo shinian97/VAST15000 --repo_type model \
        --out datasets/external/vast15000_listing.txt

A zip stores its file table (the "central directory") at the END of the archive, so the
whole listing can be read with two small range requests per file. For the 23 parts of
shinian97/VAST15000 that is roughly 25MB of transfer instead of 48GB -- which makes it
cheap to measure the FULL recoverable set rather than extrapolating from one part, as
scripts/check_zip_clip_coverage.py had to.

Writes one file name per line. Feed the result to check_zip_clip_coverage.py to get the
overlap with the clips we lack.
"""
import argparse
import io
import os
import struct
import sys
import urllib.request

EOCD_SIG = b'PK\x05\x06'
EOCD64_LOCATOR_SIG = b'PK\x06\x07'
CEN_SIG = b'PK\x01\x02'


def http_range(url, start, end, token=None):
    req = urllib.request.Request(url)
    req.add_header('Range', 'bytes=%d-%d' % (start, end))
    if token:
        req.add_header('Authorization', 'Bearer %s' % token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def http_size(url, token=None):
    req = urllib.request.Request(url, method='HEAD')
    if token:
        req.add_header('Authorization', 'Bearer %s' % token)
    with urllib.request.urlopen(req, timeout=60) as r:
        # HF redirects to a CDN; urllib follows it and the final response carries the length
        n = r.headers.get('Content-Length')
        return int(n) if n else None


def central_directory_names(url, token=None):
    """File names from the zip's central directory, read over HTTP ranges."""
    size = http_size(url, token)
    if not size:
        raise RuntimeError('could not determine remote size (no Content-Length)')

    tail_len = min(65536 + 22, size)
    tail = http_range(url, size - tail_len, size - 1, token)
    idx = tail.rfind(EOCD_SIG)
    if idx < 0:
        raise RuntimeError('no end-of-central-directory record found')
    eocd = tail[idx:idx + 22]
    cd_size, cd_off = struct.unpack('<II', eocd[12:20])

    # ZIP64: the 32-bit fields saturate and the real values live in a separate record.
    # Refusing is better than silently listing a truncated table.
    if cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
        if tail.rfind(EOCD64_LOCATOR_SIG) < 0:
            raise RuntimeError('ZIP64 sizes but no ZIP64 locator -- refusing to guess')
        raise RuntimeError('ZIP64 archive; this reader handles classic zips only')

    cd = http_range(url, cd_off, cd_off + cd_size - 1, token)
    names, pos = [], 0
    while pos + 46 <= len(cd):
        if cd[pos:pos + 4] != CEN_SIG:
            break
        n_len, x_len, c_len = struct.unpack('<HHH', cd[pos + 28:pos + 34])
        name = cd[pos + 46:pos + 46 + n_len].decode('utf-8', 'replace')
        names.append(name)
        pos += 46 + n_len + x_len + c_len
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--repo_type', default='model', choices=['model', 'dataset'])
    ap.add_argument('--files', nargs='*', help='specific files; default = every .zip in the repo')
    ap.add_argument('--out', default='datasets/external/remote_zip_listing.txt')
    ap.add_argument('--token', default=os.environ.get('HF_TOKEN'))
    args = ap.parse_args()

    files = args.files
    if not files:
        try:
            from huggingface_hub import list_repo_files
        except ImportError:
            print('FATAL: pass --files, or install huggingface_hub to enumerate them',
                  file=sys.stderr)
            return 2
        files = [f for f in list_repo_files(args.repo, repo_type=args.repo_type)
                 if f.lower().endswith('.zip')]
    print('%d zip file(s) in %s' % (len(files), args.repo))

    base = 'https://huggingface.co/%s%s/resolve/main/' % (
        '' if args.repo_type == 'model' else 'datasets/', args.repo)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total, failed = 0, []
    with open(out_path, 'w') as fh:
        for f in sorted(files):
            url = base + f
            try:
                names = central_directory_names(url, args.token)
            except Exception as exc:                      # noqa: BLE001 - report and continue
                print('  %-52s FAILED: %s' % (f, exc))
                failed.append(f)
                continue
            vids = [n for n in names if n.lower().endswith('.mp4')]
            print('  %-52s %6d entries (%d mp4)' % (f, len(names), len(vids)))
            fh.write('\n'.join(names) + '\n')
            total += len(vids)

    print('\n%d mp4 entries listed -> %s' % (total, args.out))
    if failed:
        print('%d archive(s) could not be read: %s' % (len(failed), ', '.join(failed)))
        print('Counts below are a LOWER BOUND -- those archives are unaccounted for.')
    print('\nNext: python3 scripts/check_zip_clip_coverage.py %s' % args.out)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
