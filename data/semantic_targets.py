import os
import json
import argparse
import torch
import torch.nn.functional as F

# S* semantic-target cache (plan item 1.3.5) -- the ONLY new data artifact SCA needs.
# Built ONCE per dataset from the captions in its annotation json (e.g. annotations150k.json ->
# s_star_150k.pt) with a FROZEN sentence-embedding model; training only gathers cached rows.
# Sparsified to per-row top-k so the 150k x 150k affinity never materialises.

# alternative-runtime weight copies never fetched NOR required from the cache; MUST match
# what scripts/prefetch_models.py skips (it imports this constant)
HF_IGNORE_PATTERNS = ['onnx/*', 'openvino/*', '*.onnx', '*.h5', 'tf_*', 'flax_*', '*.msgpack']

_ID_KEYS = ('video_id', 'clip_id', 'id', 'image_id', 'audio_id')
# priority order; 'vast_cap' is VAST-27M's omni caption -- the text the model trains
# against on vast27m_150k, hence the right S* source there
_CAP_KEYS = ('desc', 'caption', 'text', 'raw_caption', 'vast_cap')


def _read_annotations(path, caption_key=None):
    """Strict annotation reader. The id key and caption key are resolved ONCE (from the
    first item, by the priority in _ID_KEYS/_CAP_KEYS, or forced via caption_key) and then
    REQUIRED on every item -- no per-item guessing, no skipping, no index-substituted ids:
    any of those would produce an S* cache that mismatches the training batches in ways
    that only surface (or worse, don't) much later. Returns (ids, caps, id_key, cap_key)
    so the builder can record the resolved keys in the cache meta."""
    with open(path) as f:
        anno = json.load(f)
    if isinstance(anno, dict):
        if 'data' not in anno:
            raise ValueError(
                f'{path}: dict-shaped annotation file without a "data" key '
                f'(top-level keys: {list(anno)[:8]}) -- refusing to guess the item list.')
        anno = anno['data']
    if not isinstance(anno, list) or not anno:
        raise ValueError(f'{path}: expected a non-empty list of annotation items')

    first = anno[0]
    id_key = next((k for k in _ID_KEYS if k in first), None)
    if caption_key is not None:
        cap_key = caption_key if caption_key in first else None
    else:
        cap_key = next((k for k in _CAP_KEYS if k in first), None)
    if id_key is None or cap_key is None:
        want = f'caption key {caption_key!r}' if caption_key else f'caption keys {_CAP_KEYS}'
        raise ValueError(
            f'{path}: item 0 lacks {"an id" if id_key is None else "a caption"} '
            f'(keys present: {sorted(first)[:10]}; accepted id keys {_ID_KEYS}, {want}) '
            f'-- pick the right field via --caption_key if the schema differs.')

    ids, caps = [], []
    for i, item in enumerate(anno):
        if id_key not in item or cap_key not in item:
            raise ValueError(
                f'{path}: item {i} lacks {id_key!r} or {cap_key!r} (resolved from item 0; '
                f'keys present: {sorted(item)[:10]}) -- inconsistent schema, refusing.')
        cap = item[cap_key]
        if isinstance(cap, list):
            cap = cap[0]
        ids.append(str(item[id_key]))
        caps.append(cap)
    return ids, caps, id_key, cap_key


def _resolve_sbert_model(model_name):
    """Resolve the model to a LOCAL snapshot directory from the HF cache whenever one
    exists. sentence-transformers 2.2.x ignores HF_HUB_OFFLINE and phones home from its
    own snapshot_download, so on offline compute nodes it must be handed a filesystem
    path, never a hub name. Online with no cached copy: return the name unchanged (the
    normal first fetch on a login node)."""
    if os.path.isdir(model_name):
        return model_name
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo_id=model_name, local_files_only=True,
                                 ignore_patterns=HF_IGNORE_PATTERNS)
    except Exception as e:
        offline = (os.environ.get('HF_HUB_OFFLINE') == '1'
                   or os.environ.get('TRANSFORMERS_OFFLINE') == '1')
        if offline:
            raise RuntimeError(
                f'S* builder: {model_name!r} is not in the local HF cache '
                f'(HF_HOME={os.environ.get("HF_HOME", "~/.cache/huggingface")}) and this '
                'node is OFFLINE. Run scripts/prefetch_models.py on a login node first.'
            ) from e
        return model_name


@torch.no_grad()
def _embed_captions(captions, model_name, device, batch_size=256, impl='sbert'):
    """Frozen sentence embeddings. impl is EXPLICIT, never inferred from what happens to
    be installed: 'sbert' (default; the model's own pooling stack) or 'hf' (transformers
    AutoModel + mean pooling -- NOT numerically identical for models with extra pooling/
    dense layers). Two S* caches built with different impls are different targets, so the
    choice is recorded in the cache meta and a missing package is a hard error."""
    if impl == 'sbert':
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                'S* builder: sentence-transformers is not installed. Install it '
                '(pip install sentence-transformers) or EXPLICITLY pass --embedding_impl '
                'hf -- refusing to switch implementations silently, the resulting S* '
                'would differ.') from e
        model = SentenceTransformer(_resolve_sbert_model(model_name), device=device)
        emb = model.encode(captions, batch_size=batch_size, convert_to_tensor=True,
                           normalize_embeddings=True, show_progress_bar=True)
        return emb.float().cpu()
    elif impl == 'hf':
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
    raise ValueError(f'unknown embedding impl {impl!r} (sbert | hf)')


@torch.no_grad()
def build_semantic_targets(annotation_json, out_path,
                           model_name='sentence-transformers/all-mpnet-base-v2',
                           tau_star=0.5, topk=64, threshold=0.0,
                           batch_size=256, chunk=2048, device=None, embedding_impl='sbert',
                           caption_key=None):
    """Compute and cache the sparsified S* affinity.

    Affinity: s*_ij = ((cos_ij + 1) / 2) ** (1 / tau_star)  in [0, 1]; tau_star < 1 sharpens
    (A9 sweeps model_name / tau_star / topk / threshold). Per row, only the top-k entries
    >= threshold are stored (indices + values); everything else is treated as 0 downstream.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ids, caps, id_key, cap_key = _read_annotations(annotation_json, caption_key=caption_key)
    print(f'[S*] {len(ids)} items; id key {id_key!r}, caption key {cap_key!r}', flush=True)
    emb = _embed_captions(caps, model_name, device, batch_size, impl=embedding_impl)
    if len(ids) != len(set(ids)):
        # multi-caption annotations (e.g. MSR-VTT train: 20 caption rows per video) list one
        # row per caption; the clip-level S* embedding is the L2-normalised mean of that
        # clip's caption embeddings (normalising the sum == normalising the mean)
        uniq_ids = list(dict.fromkeys(ids))
        row = {v: i for i, v in enumerate(uniq_ids)}
        agg = torch.zeros(len(uniq_ids), emb.shape[1])
        agg.index_add_(0, torch.tensor([row[v] for v in ids]), emb)
        emb = F.normalize(agg, dim=-1)
        print(f'[S*] {len(ids)} captions over {len(uniq_ids)} unique ids '
              f'-> per-id mean caption embedding', flush=True)
        ids = uniq_ids

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
                      'threshold': threshold, 'embedding_impl': embedding_impl,
                      'id_key': id_key, 'caption_key': cap_key,
                      'annotation_json': os.path.abspath(annotation_json)}}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(cache, out_path)
    return cache


class SemanticTargets:
    """Loads an S* cache and serves dense (B, B) target blocks for a training batch.

    gather(ids) intersects each batch row's stored top-k list with the batch itself; pairs
    outside the stored top-k are 0, the diagonal is 1. An id missing from the cache is a HARD
    ERROR: it means the cache was built from a different annotation file than the one being
    trained on, and silently substituting a one-hot row would corrupt every L_sem batch."""

    def __init__(self, cache_path):
        cache = torch.load(cache_path, map_location='cpu')
        self.row_of = {vid: i for i, vid in enumerate(cache['ids'])}
        self.topk_idx = cache['topk_idx']
        self.topk_val = cache['topk_val'].float()
        self.meta = cache.get('meta', {})

    def gather(self, ids, device=None):
        B = len(ids)
        unknown = [str(i) for i in ids if str(i) not in self.row_of]
        if unknown:
            raise KeyError(
                f'S* cache has no rows for batch ids {unknown[:5]}'
                f'{" ..." if len(unknown) > 5 else ""} ({len(unknown)}/{B} missing). The cache '
                f'was built from {self.meta.get("annotation_json", "<unknown>")} -- rebuild it '
                'from the annotation file this run actually trains on.')
        rows = [self.row_of[str(i)] for i in ids]
        s = torch.zeros(B, B)
        col_of = {r: j for j, r in enumerate(rows)}
        for a, r in enumerate(rows):
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
    p.add_argument('--embedding_impl', default='sbert', choices=['sbert', 'hf'],
                   help='explicit embedding implementation; recorded in the cache meta')
    p.add_argument('--caption_key', default=None,
                   help='force a specific caption field (default: priority auto-resolve, '
                        'e.g. vast_cap on VAST-27M); recorded in the cache meta')
    args = p.parse_args()
    cache = build_semantic_targets(args.annotation_json, args.out_path, args.model_name,
                                   args.tau_star, args.topk, args.threshold,
                                   args.batch_size, args.chunk,
                                   embedding_impl=args.embedding_impl,
                                   caption_key=args.caption_key)
    print(f"S* cache: {len(cache['ids'])} rows, top-{cache['topk_idx'].shape[1]} -> {args.out_path}")


if __name__ == '__main__':
    main()
