"""Đánh giá retrieval (chưa dùng LLM): Hit@1, Evidence Hit@3, Context Evidence Rate trên một split.

Câu thiếu bằng chứng vẫn được chạy và lưu trace, nhưng KHÔNG vào mẫu số retrieval metrics.
"""
import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from context import NeighborExpander

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from baseline import pack_context                                   # cách đóng gói context của starter
from retrievers import BM25Retriever, DenseRetriever, HybridRRF, load_chunks


def squash(text):
    return re.sub(r'\s+', ' ', text).strip()


def covers(chunks, g):
    """Một phần evidence được phủ khi có chunk đúng nguồn + trang chứa trọn nó."""
    return any(c['source_id'] == g['source_id'] and c['page'] == g['page']
               and squash(g['evidence']) in squash(c['text']) for c in chunks)


def is_hit(q, chunks):
    """evidence_mode 'any': một evidence thay thế là đủ; 'all': mọi phần bắt buộc đều phải có."""
    check = all if q.get('evidence_mode') == 'all' else any
    return check(covers(chunks, g) for g in q['gold'])


def build(names, chunks):
    made = {}
    if {'dense', 'hybrid_rrf'} & set(names):
        made['dense'] = DenseRetriever(chunks)                      # load model 1 lần, dùng chung
    if {'bm25', 'hybrid_rrf'} & set(names):
        made['bm25'] = BM25Retriever(chunks)
    if 'hybrid_rrf' in names:
        made['hybrid_rrf'] = HybridRRF([made['dense'], made['bm25']])
    return {n: made[n] for n in names}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True, choices=('dev', 'test'))
    p.add_argument('--methods', nargs='+', default=['dense', 'bm25', 'hybrid_rrf'])
    p.add_argument('--top-k', type=int, default=3, help='số chunk đưa vào context')
    p.add_argument('--budget', type=int, default=2000, help='context budget, đơn vị ký tự, gồm header nguồn')
    p.add_argument('--no-filter', action='store_true', help='bỏ filter người dùng cung cấp')
    p.add_argument('--neighbors', type=int, default=0, help='số chunk lân cận mỗi phía thêm vào context')
    p.add_argument('--tag', default='baseline', help='tên cấu hình, dùng làm tên thư mục kết quả')
    a = p.parse_args()
    if a.split == 'test' and not (ROOT / 'results' / 'selection.json').exists():
        sys.exit('Chưa chốt cấu hình trên dev (results/selection.json) -> không chạy test.')

    questions = [q for q in json.loads((ROOT / 'data' / 'day7' / 'questions.json').read_text(encoding='utf-8'))
                 if q['split'] == a.split]
    chunks = load_chunks()
    retrievers = build(a.methods, chunks)
    fetch = max(a.top_k, 3)                       # luôn lấy đủ top-3 để so Hit@3 cùng cutoff

    traces, summary = [], {}
    expander = NeighborExpander(chunks, a.neighbors) if a.neighbors else None
    for name, r in retrievers.items():
        rows = []
        for q in questions:
            filters = None if a.no_filter else q.get('filter')
            t0 = time.perf_counter()
            hits = r.search(q['question'], filters=filters, top_k=fetch)
            ms = (time.perf_counter() - t0) * 1000
            selected = expander.expand(hits[:a.top_k]) if expander else hits[:a.top_k]
            context, ids = pack_context(selected, budget=a.budget)
            in_context = [h for h in selected if h['id'] in ids]
            row = {'method': name, 'tag': a.tag, 'query_id': q['id'], 'split': q['split'],
                   'question': q['question'], 'type': q.get('type'), 'answerable': q['answerable'],
                   'filter': filters, 'top_k': a.top_k, 'budget_chars': a.budget, 'neighbors': a.neighbors,
                   'results': [{'rank': h['rank'], 'id': h['id'], 'page': h['page'], 'score': round(h['score'], 5),
                                **({'component_ranks': h['component_ranks']} if 'component_ranks' in h else {})}
                               for h in hits],
                   'context_ids': ids, 'context_chars': len(context), 'elapsed_ms': round(ms, 1)}
            if q['answerable']:
                row.update(hit1=is_hit(q, hits[:1]), hit3=is_hit(q, hits[:3]), context_hit=is_hit(q, in_context))
            rows.append(row)
        scored = [x for x in rows if x['answerable']]
        n = len(scored)
        summary[name] = {
            'N': n,
            'hit1': sum(x['hit1'] for x in scored), 'hit3': sum(x['hit3'] for x in scored),
            'context_hit': sum(x['context_hit'] for x in scored),
            'avg_context_chars': round(statistics.mean(x['context_chars'] for x in scored)),
            'median_ms': round(statistics.median(x['elapsed_ms'] for x in rows), 1),
            'cost_usd': 0,
            'miss_hit3': [x['query_id'] for x in scored if not x['hit3']],
        }
        traces += rows

    out = ROOT / 'results' / 'retrieval' / f'{a.split}-{a.tag}'
    out.mkdir(parents=True, exist_ok=True)
    meta = {'run_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'split': a.split,
            'tag': a.tag, 'top_k': a.top_k, 'budget_chars': a.budget, 'filter': not a.no_filter, 'neighbors': a.neighbors,
            'index': 'A (Day 6, fixed 500/50)', 'embedding': 'google/embeddinggemma-2, 768d',
            'bm25': {'k1': 1.5, 'b': 0.75}, 'rrf': {'k': 60, 'candidates': 20},
            'hit_definition': 'whitespace-normalized substring, đúng source_id + page; evidence_mode any/all'}
    for fname, obj in (('traces.json', traces), ('summary.json', {'config': meta, 'results': summary})):
        (out / fname).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(f"\n{a.split} | tag={a.tag} | top_k={a.top_k} | budget={a.budget} | filter={not a.no_filter}")
    print(f"{'method':12}{'Hit@1':>9}{'Hit@3':>9}{'Context':>9}{'ms':>8}  miss@3")
    for name, s in summary.items():
        print(f"{name:12}{s['hit1']:>5}/{s['N']:<3}{s['hit3']:>5}/{s['N']:<3}{s['context_hit']:>5}/{s['N']:<3}"
              f"{s['median_ms']:>8}  {s['miss_hit3']}")
    print(f'-> {out}')


if __name__ == '__main__':
    main()
