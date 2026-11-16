"""Ba phương pháp retrieval trên cùng corpus (index A Day 6): Dense, BM25, Hybrid RRF.

Mọi retriever cùng một interface: search(query, *, filters, top_k) -> list[dict]
với id, text, source_id, source, page, metadata, score, rank, strategy.
Lỗi DB/model được raise, không có fallback giả.
"""
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_FILTERS = {'chapter'}          # filter chỉ đến từ thông tin người dùng cung cấp


def check_filters(filters):
    for key in filters or {}:
        if key not in ALLOWED_FILTERS:
            raise ValueError(f'filter "{key}" không được hỗ trợ; chỉ có {sorted(ALLOWED_FILTERS)}')


def matches(chunk, filters):
    return all(chunk['metadata'].get(k) == v for k, v in (filters or {}).items())


def result(chunk, score, rank, strategy, **extra):
    return {'id': chunk['id'], 'text': chunk['text'], 'source_id': chunk['source_id'],
            'source': chunk['source'], 'page': chunk['page'], 'metadata': chunk['metadata'],
            'score': float(score), 'rank': rank, 'strategy': strategy, **extra}


def load_chunks(path=ROOT / 'data' / 'day7' / 'chunks.json'):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ---------------------------------------------------------------- 1. Dense
class DenseRetriever:
    """Query embedding (embeddinggemma-2, prompt QuestionAnswering) -> pgvector cosine, index A."""
    name = 'dense'

    def __init__(self, chunks, embedder=None, index='A'):
        # Import tại đây: BM25/Hybrid/test chạy được mà không cần thư viện DB và model.
        import psycopg
        from dotenv import load_dotenv
        from embeddings import GemmaEmbedder
        self._connect = psycopg.connect
        load_dotenv(ROOT / 'starter' / '.env')
        self.url = os.environ['DATABASE_URL']
        self.index = index
        self.embedder = embedder or GemmaEmbedder()
        self.by_id = {c['id']: c for c in chunks}   # lấy text/metadata y hệt bản BM25 dùng

    def search(self, query, *, filters=None, top_k=3):
        check_filters(filters)
        qv = '[' + ','.join(f'{x:.7g}' for x in self.embedder.embed_query(query)) + ']'
        where, params = ['strategy = %s'], [self.index]
        if filters and 'chapter' in filters:
            where.append('chapter = %s')
            params.append(filters['chapter'])
        sql = (f"SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score FROM chunks "
               f"WHERE {' AND '.join(where)} ORDER BY embedding <=> %s::vector LIMIT %s")
        with self._connect(self.url, connect_timeout=5) as conn:
            rows = conn.execute(sql, [qv, *params, qv, top_k]).fetchall()
        missing = [cid for cid, _ in rows if cid not in self.by_id]
        if missing:
            raise RuntimeError(f'DB và chunks.json lệch phiên bản: {missing[:3]}; chạy lại export_day6.py')
        return [result(self.by_id[cid], s, i + 1, self.name) for i, (cid, s) in enumerate(rows)]


# ---------------------------------------------------------------- 2. BM25
def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


class BM25Retriever:
    """BM25 Okapi tự cài (thư viện chuẩn), k1=1.5, b=0.75; tách từ đơn giản theo \\w+."""
    name = 'bm25'

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokenize(c['text'])) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tf]
        self.avgdl = sum(self.lens) / len(self.lens)
        n = len(chunks)
        df = Counter(term for tf in self.tf for term in tf)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query, *, filters=None, top_k=3):
        check_filters(filters)
        terms = tokenize(query)
        scored = []
        for chunk, tf, dl in zip(self.chunks, self.tf, self.lens):
            if not matches(chunk, filters):
                continue
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scored.append((s, chunk))
        scored.sort(key=lambda x: (-x[0], x[1]['id']))       # hòa điểm -> xếp theo id cho ổn định
        return [result(c, s, i + 1, self.name) for i, (s, c) in enumerate(scored[:top_k])]


# ---------------------------------------------------------------- 3. Hybrid RRF
class HybridRRF:
    """Reciprocal Rank Fusion: gộp theo thứ hạng, không cộng score khác thang đo."""
    name = 'hybrid_rrf'

    def __init__(self, retrievers, k=60, candidates=20):
        self.retrievers, self.k, self.candidates = retrievers, k, candidates

    def search(self, query, *, filters=None, top_k=3):
        fused, chunk_of, ranks = {}, {}, {}
        for r in self.retrievers:
            for hit in r.search(query, filters=filters, top_k=self.candidates):
                fused[hit['id']] = fused.get(hit['id'], 0) + 1 / (self.k + hit['rank'])
                chunk_of[hit['id']] = hit
                ranks.setdefault(hit['id'], {})[r.name] = hit['rank']
        order = sorted(fused, key=lambda cid: (-fused[cid], cid))[:top_k]
        return [result(chunk_of[cid], fused[cid], i + 1, self.name, component_ranks=ranks[cid])
                for i, cid in enumerate(order)]


if __name__ == '__main__':
    # Thử nhanh: python -X utf8 src/retrievers.py "câu hỏi"
    query = ' '.join(sys.argv[1:]) or 'What is pseudonymization?'
    chunks = load_chunks()
    dense, bm25 = DenseRetriever(chunks), BM25Retriever(chunks)
    for r in (dense, bm25, HybridRRF([dense, bm25])):
        print(f'\n== {r.name}')
        for h in r.search(query, top_k=3):
            print(f"  #{h['rank']} {h['score']:.4f} p{h['page']} {h['id']} {h.get('component_ranks', '')}")
            print('     ', re.sub(r'\s+', ' ', h['text'])[:110])
