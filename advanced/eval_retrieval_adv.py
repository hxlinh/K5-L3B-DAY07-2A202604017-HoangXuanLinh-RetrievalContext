"""Bước A5: so sánh các phương pháp retrieval ở MỨC TRANG trên benchmark nâng cao (không gọi API).

Phương pháp (advanced/page_retriever.py): text_dense, bm25, image, interleaved,
hybrid_text = RRF(text_dense, bm25), hybrid_image = + image, hybrid_inter = + interleaved.

Hit@k (trang): câu thường đạt khi có >= 1 trang evidence trong top-k; câu multi-hop cần MỌI trang evidence
trong top-k. Câu thiếu bằng chứng không vào mẫu số.
Chạy: python -X utf8 advanced/eval_retrieval_adv.py --split dev
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from page_retriever import CANDIDATES, METHODS, RRF_K, PageRetriever

KS = (1, 3, 5)


def hit(q, ranking, k):
    pages = {e['page'] for e in q['evidence']}
    top = set(ranking[:k])
    return pages <= top if q['group'] == 'multihop' else bool(pages & top)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True, choices=('dev', 'test'))
    p.add_argument('--tag', default='baseline')
    a = p.parse_args()
    if a.split == 'test' and not (HERE / 'results' / 'selection.json').exists():
        sys.exit('Chưa chốt cấu hình trên dev (advanced/results/selection.json) -> không chạy test.')

    questions = [q for q in json.loads((HERE / 'benchmark' / 'questions.json').read_text(encoding='utf-8'))
                 if q['split'] == a.split]
    retriever = PageRetriever()
    t0 = time.perf_counter()
    qvecs = retriever.model.encode([q['question'] for q in questions], prompt_name='QuestionAnswering',
                                   normalize_embeddings=True)
    embed_ms = (time.perf_counter() - t0) * 1000 / len(questions)

    traces = []
    for q, v in zip(questions, qvecs):
        r = retriever.rank_all(q['question'], v)
        row = {'query_id': q['id'], 'group': q['group'], 'question': q['question'],
               'evidence_pages': sorted({e['page'] for e in q.get('evidence', [])}),
               'rankings': {m: x[:10] for m, x in r.items()}}
        if q['group'] != 'unanswerable':
            row['hits'] = {m: {f'@{k}': hit(q, x, k) for k in KS} for m, x in r.items()}
        traces.append(row)

    scored = [t for t in traces if 'hits' in t]
    groups = sorted({t['group'] for t in scored})
    summary = {}
    for m in METHODS:
        summary[m] = {'overall': {f'@{k}': f"{sum(t['hits'][m][f'@{k}'] for t in scored)}/{len(scored)}" for k in KS}}
        for g in groups:
            rows = [t for t in scored if t['group'] == g]
            summary[m][g] = f"{sum(t['hits'][m]['@3'] for t in rows)}/{len(rows)}"

    out = HERE / 'results' / 'retrieval' / f'{a.split}-{a.tag}'
    out.mkdir(parents=True, exist_ok=True)
    meta = {'run_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'split': a.split,
            'query_prompt': 'QuestionAnswering', 'rrf_k': RRF_K, 'candidates': CANDIDATES,
            'hit_definition': 'page-level; multihop needs all evidence pages in top-k; unanswerable excluded',
            'query_embed_ms': round(embed_ms, 1)}
    (out / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps({'config': meta, 'results': summary}, ensure_ascii=False, indent=2),
                                      encoding='utf-8')

    print(f"\n{a.split}: {len(scored)} câu có đáp án | Hit@k mức trang | cột nhóm = Hit@3")
    print(f"{'method':14}{'@1':>7}{'@3':>7}{'@5':>7}   " + '  '.join(f'{g:>9}' for g in groups))
    for m in METHODS:
        s = summary[m]
        print(f"{m:14}{s['overall']['@1']:>7}{s['overall']['@3']:>7}{s['overall']['@5']:>7}   "
              + '  '.join(f'{s[g]:>9}' for g in groups))
    print('->', out)


if __name__ == '__main__':
    main()
