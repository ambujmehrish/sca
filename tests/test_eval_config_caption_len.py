"""Paragraph-retrieval benchmarks must not truncate their queries.

DiDeMo and ActivityNet retrieval concatenates a video's segment captions into one query,
which routinely exceeds the default max_caption_len of 40 tokens. Every config that shipped
with this repo sets 70 for those two benchmarks; the configs generated for the evaluation
campaign inherited the default instead, silently discarding roughly a third of each query
on exactly the two benchmarks where our numbers sat 4-7 R@1 below the published ones -- for
our GRAM reproduction as well as for SCA. This test fails if any config regresses.
"""
import glob
import json
import os
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PARAGRAPH_BENCHMARKS = ('didemo', 'activitynet')
REQUIRED = 70


class TestParagraphCaptionLen(unittest.TestCase):
    def test_paragraph_benchmarks_are_not_truncated(self):
        offenders = []
        for pattern in ('benchmark_eval/configs*/*.json', 'config/*/*/*.json'):
            for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
                try:
                    cfg = json.load(open(path))
                except (ValueError, IOError):
                    continue
                data = json.dumps(cfg.get('data_cfg', {})).lower()
                if not any(b in data for b in PARAGRAPH_BENCHMARKS):
                    continue
                got = cfg.get('model_cfg', {}).get('max_caption_len')
                if got != REQUIRED:
                    offenders.append((os.path.relpath(path, ROOT), got))
        self.assertEqual(
            offenders, [],
            'these DiDeMo/ActivityNet configs would truncate paragraph queries '
            '(need max_caption_len=%d): %s' % (REQUIRED, offenders))


if __name__ == '__main__':
    unittest.main()
