"""Tìm kiếm mức TRANG trên index nâng cao; dùng chung cho đánh giá retrieval và agent.

Phương pháp: text_dense, bm25, image, interleaved và các tổ hợp RRF (hybrid_text, hybrid_image, hybrid_inter).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from retrievers import BM25Retriever

INDEX = HERE / 'data' / 'index'
RRF_K, CANDIDATES = 60, 30
HYBRIDS = {'hybrid_text': ('text_dense', 'bm25'),
           'hybrid_image': ('text_dense', 'bm25', 'image'),
           'hybrid_inter': ('text_dense', 'bm25', 'interleaved')}
METHODS = ['text_dense', 'bm25', 'image', 'interleaved', *HYBRIDS]


def rrf(*rankings):
    score = defaultdict(float)
    for ranking in rankings:
        for r, page in enumerate(ranking, 1):
            score[page] += 1 / (RRF_K + r)
    return sorted(score, key=lambda p: (-score[p], p))[:CANDIDATES]


def pages_from_chunks(chunk_pages, order, limit=CANDIDATES):
    """Thứ tự chunk -> thứ tự trang (giữ lần xuất hiện đầu tiên)."""
    seen, out = set(), []
    for i in order:
        if chunk_pages[i] not in seen:
            seen.add(chunk_pages[i])
            out.append(chunk_pages[i])
            if len(out) == limit:
                break
    return out


class PageRetriever:
    def __init__(self, model=None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.chunks = [json.loads(l) for l in (INDEX / 'text_chunks.jsonl').read_text(encoding='utf-8').splitlines()]
        self.chunk_pages = [c['page'] for c in self.chunks]
        self.position = {c['id']: i for i, c in enumerate(self.chunks)}
        self.text_vec = np.load(INDEX / 'text_vectors.npy')
        self.page_vecs = {k: (json.loads((INDEX / f'{k}_pages.json').read_text(encoding='utf-8')),
                              np.load(INDEX / f'{k}_vectors.npy')) for k in ('image', 'interleaved')}
        self.bm25 = BM25Retriever([{'id': c['id'], 'text': c['text'], 'source_id': 'pods',
                                    'source': 'Principles-of-Data-Science-WEB.pdf', 'page': c['page'],
                                    'metadata': {'chapter': c['chapter']}} for c in self.chunks])
        self.model = model or SentenceTransformer('google/embeddinggemma-2', config_kwargs={'audio_config': None},
                                                  model_kwargs={'dtype': torch.float32})
        self.page_text = defaultdict(str)
        for c in self.chunks:                       # ghép lại text trang (bỏ phần chồng lấn 50 ký tự)
            self.page_text[c['page']] += c['text'] if not self.page_text[c['page']] else c['text'][50:]

    def rank_all(self, query, qvec=None):
        """Mọi phương pháp cùng lúc -> {method: [trang xếp hạng]}; embed query đúng một lần."""
        if qvec is None:
            qvec = self.model.encode([query], prompt_name='QuestionAnswering', normalize_embeddings=True)[0]
        r = {'text_dense': pages_from_chunks(self.chunk_pages, np.argsort(-(self.text_vec @ qvec))),
             'bm25': pages_from_chunks(self.chunk_pages,
                                       [self.position[h['id']] for h in self.bm25.search(query, top_k=200)])}
        for kind, (pages, vec) in self.page_vecs.items():
            r[kind] = [pages[i] for i in np.argsort(-(vec @ qvec))[:CANDIDATES]]
        for name, parts in HYBRIDS.items():
            r[name] = rrf(*(r[p] for p in parts))
        return r

    def search(self, query, method, top_k=3):
        if method not in METHODS:
            raise ValueError(f'method phải là một trong {METHODS}')
        return self.rank_all(query)[method][:top_k]
