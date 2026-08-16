#!/usr/bin/env python3
"""A10 Stage-0 data preparation: REAL image-text data (Flickr8k) + frozen encoders.

Downloads jxie/flickr8k (the standard Flickr8k with 5 human captions per image, 6k/1k/1k
splits), encodes images and captions with a FROZEN CLIP ViT-B/32, writes per-split feature
tensors, and builds the S* caches with data/semantic_targets.build_semantic_targets -- the
exact production code path (frozen sentence-embedding affinities, tau* sharpening, top-k
sparsification, disk cache).

This is the plan's P1 "image-text smoke (current experiments = Stage-0 validation)": k=2
(text + one gallery modality), frozen backbones, trainable projection heads only.

    python3 experiments/a10_prepare_flickr8k.py --workdir experiments/a10_workdir
"""
import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CLIP_NAME = 'openai/clip-vit-base-patch32'
SBERT_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
N_CAPTIONS = 5


def _features(out):
    """transformers <5 returns the projected features tensor directly; >=5 returns the model
    output object with the projected features in .pooler_output."""
    return out if torch.is_tensor(out) else out.pooler_output


@torch.no_grad()
def encode_split(ds, clip, processor, device, batch_size=64):
    from PIL import Image  # noqa: F401  (datasets decodes to PIL)
    img_feats, txt_feats = [], []
    n = len(ds)
    for s in range(0, n, batch_size):
        rows = ds[s:s + batch_size]
        images = [im.convert('RGB') for im in rows['image']]
        pix = processor(images=images, return_tensors='pt').to(device)
        img_feats.append(_features(clip.get_image_features(**pix)).float().cpu())
        caps = []
        for c in range(N_CAPTIONS):
            caps += rows[f'caption_{c}']
        tok = processor.tokenizer(caps, padding=True, truncation=True, max_length=77,
                                  return_tensors='pt').to(device)
        t = _features(clip.get_text_features(**tok)).float().cpu()   # (5*B, d) caption-major
        txt_feats.append(t.reshape(N_CAPTIONS, len(images), -1).permute(1, 0, 2))  # (B, 5, d)
        if (s // batch_size) % 10 == 0:
            print(f'  encoded {s + len(images)}/{n}', flush=True)
    return torch.cat(img_feats), torch.cat(txt_feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default='experiments/a10_workdir')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--tau_star', type=float, default=0.5)
    ap.add_argument('--topk', type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from datasets import load_dataset
    from transformers import CLIPModel, CLIPProcessor
    from data.semantic_targets import build_semantic_targets

    print(f'[prepare] loading flickr8k (real data) on {device}', flush=True)
    # Prefer the prefetch manifest of local parquet shards (works on OFFLINE nodes --
    # `datasets`<2.16 cannot resolve a hub dataset offline even when cached). Same real
    # data either way; offline with no manifest is a hard error.
    manifest_path = os.path.join(os.environ.get('MODELS_DIR', ''),
                                 'jxie__flickr8k_parquet.json')
    offline = (os.environ.get('HF_HUB_OFFLINE') == '1'
               or os.environ.get('HF_DATASETS_OFFLINE') == '1')
    if os.environ.get('MODELS_DIR') and os.path.exists(manifest_path):
        from datasets import Image as HFImage
        with open(manifest_path) as f:
            data_files = json.load(f)
        missing = [p for ps in data_files.values() for p in ps if not os.path.exists(p)]
        assert not missing, f'manifest shards missing on disk: {missing[:3]}'
        dsd = load_dataset('parquet', data_files=data_files)
        dsd = dsd.cast_column('image', HFImage())          # parquet stores raw bytes
        print(f'[prepare] flickr8k from local parquet manifest ({manifest_path})', flush=True)
    elif offline:
        raise RuntimeError(
            'OFFLINE node and no local flickr8k manifest: run scripts/prefetch_models.py '
            '--models_dir $MODELS_DIR --with-smoke-data on a login node first '
            '(and export MODELS_DIR here).')
    else:
        dsd = load_dataset('jxie/flickr8k')
    clip = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)

    for split in ('train', 'test'):
        feat_path = os.path.join(args.workdir, f'features_{split}.pt')
        anno_path = os.path.join(args.workdir, f'annotations_{split}.json')
        sstar_path = os.path.join(args.workdir, f's_star_{split}.pt')
        ds = dsd[split]
        ids = [f'{split}_{i:05d}' for i in range(len(ds))]

        if not os.path.exists(feat_path):
            print(f'[prepare] encoding {split} ({len(ds)} images x {N_CAPTIONS} captions) '
                  f'with frozen {CLIP_NAME}', flush=True)
            img, txt = encode_split(ds, clip, processor, device, args.batch_size)
            torch.save({'img': img, 'txt': txt, 'ids': ids,
                        'meta': {'clip': CLIP_NAME, 'dataset': 'jxie/flickr8k',
                                 'split': split}}, feat_path)
        else:
            print(f'[prepare] {feat_path} exists, skipping encode', flush=True)

        if not os.path.exists(sstar_path):
            # the S* builder reads captions from an annotation json, exactly as on the cluster
            anno = [{'video_id': vid, 'caption': ds[i]['caption_0']}
                    for i, vid in enumerate(ids)]
            with open(anno_path, 'w') as f:
                json.dump(anno, f)
            print(f'[prepare] building S* cache for {split} via data/semantic_targets.py '
                  f'({SBERT_NAME}, tau*={args.tau_star}, topk={args.topk})', flush=True)
            build_semantic_targets(anno_path, sstar_path, model_name=SBERT_NAME,
                                   tau_star=args.tau_star, topk=args.topk, device=device)
        else:
            print(f'[prepare] {sstar_path} exists, skipping S*', flush=True)

    print('[prepare] done:', sorted(os.listdir(args.workdir)))


if __name__ == '__main__':
    main()
