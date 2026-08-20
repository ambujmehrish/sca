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

# GRAM (arXiv:2412.11959v2) Table 5, Appendix B.1. The table's columns are
# Train / Val / Test counts, then "# Frames", then "# Epochs", and its caption reads
# "# Frames refers both to training and inference." Frames are 8 for EVERY benchmark; the
# 40 and 20 that appear against DiDeMo and ActivityNet are finetuning epochs, matching the
# body text: "For finetuning we reduce the batch size to 64 and change the number of epochs
# according to the specific dataset". This constant exists so that misreading is not
# repeated: any config evaluating at something other than 8 frames fails the test.
INFERENCE_FRAMES = {'didemo': 8, 'activitynet': 8}


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


    def test_paragraph_benchmarks_use_published_inference_frames(self):
        offenders = []
        for path in sorted(glob.glob(os.path.join(ROOT, 'benchmark_eval/configs_e*/*.json'))):
            try:
                cfg = json.load(open(path))
            except (ValueError, IOError):
                continue
            data = cfg.get('data_cfg', {})
            blob = json.dumps(data).lower()
            bench = next((b for b in INFERENCE_FRAMES if b in blob), None)
            if bench is None:
                continue
            want = INFERENCE_FRAMES[bench]
            for split in ('train', 'val'):
                for entry in data.get(split, []):
                    got = entry.get('vision_sample_num')
                    if got != want:
                        offenders.append(
                            (os.path.relpath(path, ROOT), split, bench, got, want))
        self.assertEqual(
            offenders, [],
            'these configs do not use the published frame count of 8 '
            '(GRAM Tab. 5; the 40/20 in that table are finetuning epochs): %s' % offenders)


if __name__ == '__main__':
    unittest.main()


class TestNoTrainTestOverlap(unittest.TestCase):
    """A finetuning config must not train on the file it evaluates.

    config/{sca,gram}/finetune_cfg/retrieval-audiocaps.json pointed both its training split
    and its validation split at benchmark_eval/audiocaps_tva_annotation.json -- the 704-clip
    AudioCaps *test* annotation -- with training=True. Any result produced by it is trained
    on the test set. The config was inherited from the imported trunk and GRAM never uses it
    (their Tab. 5 lists AudioCaps with no finetuning epochs; their AudioCaps numbers are
    zero-shot, in their Tab. 3). Both configs now carry an explicit refusal, and this test
    fails if any finetuning config reintroduces the overlap.
    """

    def test_train_split_differs_from_val_split(self):
        offenders = []
        for path in sorted(glob.glob(os.path.join(ROOT, 'config/*/finetune_cfg/*.json'))):
            try:
                cfg = json.load(open(path))
            except (ValueError, IOError):
                continue
            data = cfg.get('data_cfg', {})
            train_files = {c.get('txt') for c in data.get('train', []) if c.get('training')}
            val_files = {c.get('txt') for c in data.get('val', [])}
            shared = train_files & val_files
            if shared:
                offenders.append((os.path.relpath(path, ROOT), sorted(shared)))
        self.assertEqual(
            offenders, [],
            'these finetuning configs train on the annotation they evaluate: %s' % offenders)
