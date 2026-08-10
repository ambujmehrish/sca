import os
import json
import argparse
import torch
import torch.nn.functional as F

# S* semantic-target cache (plan item 1.3.5) -- the ONLY new data artifact SCA needs.
# Built ONCE per dataset from the captions in its annotation json (e.g. annotations150k.json ->
# s_star_150k.pt) with a FROZEN sentence-embedding model; training only gathers cached rows.
# Sparsified to per-row top-k so the 150k x 150k affinity never materialises.

_ID_KEYS = ('video_id', 'clip_id', 'id', 'image_id', 'audio_id')
_CAP_KEYS = ('desc', 'caption', 'text', 'raw_caption')


def _read_annotations(path):
    with open(path) as f:
        anno = json.load(f)
    if isinstance(anno, dict):
        anno = anno.get('data', list(anno.values()))
    ids, caps = [], []
    for i, item in enumerate(anno):
        vid = next((item[k] for k in _ID_KEYS if k in item), i)
        cap = next((item[k] for k in _CAP_KEYS if k in item), None)
        if cap is None:
            continue
        if isinstance(cap, list):
            cap = cap[0]
        ids.append(str(vid))
        caps.append(cap)
    return ids, caps


@torch.no_grad()
def _embed_captions(captions, model_name, device, batch_size=256):
    """Frozen sentence embeddings; sentence-transformers if available, else HF mean-pooling."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        emb = model.encode(captions, batch_size=batch_size, convert_to_tensor=True,
                           normalize_embeddings=True, show_progress_bar=True)
        return emb.float().cpu()
    except ImportError:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        out = []
        for i in range(0, len(captions), batch_size):
            batch = tok(captions[i:i + batch_size], padding=True, truncation=True,
                        max_length=128, return_tensors='pt').to(device)
            h = model(**batch).last_hidden_state
            m = batch['attention_mask'].unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            out.append(F.normalize(emb, dim=-1).float().cpu())
        return torch.cat(out, dim=0)


@torch.no_grad()
def build_semantic_targets(annotation_json, out_path,
                           model_name='sentence-transformers/all-mpnet-base-v2',
                           tau_star=0.5, topk=64, threshold=0.0,
                           batch_size=256, chunk=2048, device=None):
    """Compute and cache the sparsified S* affinity.

    Affinity: s*_ij = ((cos_ij + 1) / 2) ** (1 / tau_star)  in [0, 1]; tau_star < 1 sharpens
    (A9 sweeps model_name / tau_star / topk / threshold). Per row, only the top-k entries
    >= threshold are stored (indices + values); everything else is treated as 0 downstream.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ids, caps = _read_annotations(annotation_json)
    assert len(ids) == len(set(ids)), 'duplicate ids in annotation file'
    emb = _embed_captions(caps, model_name, device, batch_size)      # (N, d), unit-norm, cpu

    N = emb.shape[0]
    k = min(topk, N)
    topk_idx = torch.empty(N, k, dtype=torch.long)
    topk_val = torch.empty(N, k)
    emb_dev = emb.to(device)
    for s in range(0, N, chunk):
        block = emb_dev[s:s + chunk] @ emb_dev.T                     # (c, N) cosine
        aff = ((block + 1.0) / 2.0).clamp(min=0.0, max=1.0) ** (1.0 / tau_star)
        v, i = aff.topk(k, dim=-1)
        if threshold > 0:
            v = v * (v >= threshold)
        topk_idx[s:s + chunk] = i.cpu()
        topk_val[s:s + chunk] = v.cpu()

    cache = {'ids': ids, 'topk_idx': topk_idx, 'topk_val': topk_val.half(),
             'meta': {'model_name': model_name, 'tau_star': tau_star, 'topk': topk,
                      'threshold': threshold, 'annotation_json': os.path.abspath(annotation_json)}}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(cache, out_path)
    return cache


class SemanticTargets:
    """Loads an S* cache and serves dense (B, B) target blocks for a training batch.

    gather(ids) intersects each batch row's stored top-k list with the batch itself; pairs
    outside the stored top-k are 0, the diagonal is 1. Unknown ids (not in the cache) fall
    back to a one-hot row (S* = I behaviour), so mixed batches never crash."""

    def __init__(self, cache_path):
        cache = torch.load(cache_path, map_location='cpu')
        self.row_of = {vid: i for i, vid in enumerate(cache['ids'])}
        self.topk_idx = cache['topk_idx']
        self.topk_val = cache['topk_val'].float()
        self.meta = cache.get('meta', {})

    def gather(self, ids, device=None):
        B = len(ids)
        rows = [self.row_of.get(str(i), -1) for i in ids]
        s = torch.zeros(B, B)
        col_of = {r: j for j, r in enumerate(rows) if r >= 0}
        for a, r in enumerate(rows):
            if r < 0:
                continue
            idx = self.topk_idx[r]
            val = self.topk_val[r]
            for i_t, v_t in zip(idx.tolist(), val.tolist()):
                b = col_of.get(i_t)
                if b is not None:
                    s[a, b] = v_t
        s.fill_diagonal_(1.0)
        return s.to(device) if device is not None else s


def main():
    p = argparse.ArgumentParser(description='Build the S* semantic-target cache from captions.')
    p.add_argument('--annotation_json', required=True)
    p.add_argument('--out_path', required=True)
    p.add_argument('--model_name', default='sentence-transformers/all-mpnet-base-v2')
    p.add_argument('--tau_star', type=float, default=0.5)
    p.add_argument('--topk', type=int, default=64)
    p.add_argument('--threshold', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--chunk', type=int, default=2048)
    args = p.parse_args()
    cache = build_semantic_targets(args.annotation_json, args.out_path, args.model_name,
                                   args.tau_star, args.topk, args.threshold,
                                   args.batch_size, args.chunk)
    print(f"S* cache: {len(cache['ids'])} rows, top-{cache['topk_idx'].shape[1]} -> {args.out_path}")


if __name__ == '__main__':
    main()
