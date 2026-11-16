"""Chạy agent thật trên một split; lưu trace (đúng format starter/agent_metrics.py) + mẫu nhãn để chấm tay."""
import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agent import SYSTEM_PROMPT, OpenAIController, run_agent
from context import NeighborExpander
from eval_retrieval import build, is_hit
from llm import OpenAIChat
from retrievers import load_chunks


def price(name):
    value = os.getenv(name)
    return float(value) if value else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True, choices=('dev', 'test'))
    p.add_argument('--method', required=True, choices=('dense', 'bm25', 'hybrid_rrf'))
    p.add_argument('--neighbors', type=int, default=0)
    p.add_argument('--max-searches', type=int, default=3)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--budget', type=int, default=2000, help='ký tự context cho MỖI lượt search')
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    if a.split == 'test' and not (ROOT / 'results' / 'selection.json').exists():
        sys.exit('Chưa chốt cấu hình trên dev (results/selection.json) -> không chạy test.')

    load_dotenv(ROOT / 'starter' / '.env')
    p_in, p_out = price('PRICE_INPUT_PER_1M'), price('PRICE_OUTPUT_PER_1M')
    questions = [q for q in json.loads((ROOT / 'data' / 'day7' / 'questions.json').read_text(encoding='utf-8'))
                 if q['split'] == a.split]
    chunks = load_chunks()
    by_id = {c['id']: c for c in chunks}
    retriever = build([a.method], chunks)[a.method]
    expander = NeighborExpander(chunks, a.neighbors) if a.neighbors else None
    llm = OpenAIChat()

    traces = []
    for q in questions:
        controller = OpenAIController(llm, a.max_searches)          # log riêng cho từng câu
        t0 = time.perf_counter()
        try:
            r = run_agent(q['question'], controller, retriever, expander, filters=q.get('filter'),
                          max_searches=a.max_searches, top_k=a.top_k, budget_per_search=a.budget)
            error = None
        except Exception as e:                                       # ghi lỗi thật, không thay bằng kết quả giả
            r = {'status': 'error', 'answer': None, 'citations': [], 'stop_reason': 'error', 'searches': None,
                 'final_context': '', 'final_context_ids': [], 'events': []}
            error = f'{type(e).__name__}: {e}'
        tin = sum(c['usage'].get('prompt_tokens', 0) for c in controller.calls)
        tout = sum(c['usage'].get('completion_tokens', 0) for c in controller.calls)
        cost = (tin * p_in + tout * p_out) / 1e6 if p_in is not None and p_out is not None else None
        ctx_hit = is_hit(q, [by_id[i] for i in r['final_context_ids']]) if q['answerable'] else None
        traces.append({'query_id': q['id'], 'split': q['split'], 'question': q['question'],
                       'answerable': q['answerable'], 'type': q.get('type'), 'filter': q.get('filter'),
                       'status': r['status'], 'answer': r['answer'], 'citations': r['citations'],
                       'stop_reason': r['stop_reason'], 'searches': r['searches'], 'error': error,
                       'context_evidence': ctx_hit, 'final_context_ids': r['final_context_ids'],
                       'final_context': r['final_context'], 'events': r['events'], 'model_calls': controller.calls,
                       'llm_calls': len(controller.calls), 'tokens_in': tin, 'tokens_out': tout,
                       'cost_usd': cost, 'elapsed_s': round(time.perf_counter() - t0, 2)})
        t = traces[-1]
        print(f"{q['id']:4} {t['status']:9} {t['stop_reason']:28} searches={t['searches']} "
              f"ctx={ctx_hit} tok={tin}+{tout} {error or ''}")

    ok = [t for t in traces if t['status'] != 'error']
    ans = [t for t in ok if t['answerable']]
    summary = {
        'config': {'run_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'split': a.split,
                   'tag': a.tag, 'method': a.method, 'neighbors': a.neighbors, 'max_searches': a.max_searches,
                   'top_k': a.top_k, 'budget_per_search': a.budget, 'model': llm.model,
                   'price_per_1m': {'input': p_in, 'output': p_out}, 'system_prompt': SYSTEM_PROMPT},
        'questions': len(traces), 'errors': len(traces) - len(ok),
        'status': dict(Counter(t['status'] for t in traces)),
        'stop_reason': dict(Counter(t['stop_reason'] for t in traces)),
        'tool_use': f"{sum(bool(t['searches']) for t in ok)}/{len(ok)}",
        'searches_distribution': dict(sorted(Counter(t['searches'] for t in ok).items())),
        'context_evidence_rate': f"{sum(t['context_evidence'] for t in ans)}/{len(ans)}",
        'tokens_in': sum(t['tokens_in'] for t in traces), 'tokens_out': sum(t['tokens_out'] for t in traces),
        'cost_usd': (sum(t['cost_usd'] for t in traces) if p_in is not None and p_out is not None else None),
        'median_s': statistics.median(t['elapsed_s'] for t in traces),
    }
    reviews = []
    for t in traces:
        if t['status'] == 'error':                 # lần chạy lỗi không cần nhãn (agent_metrics báo riêng)
            continue
        label = {'query_id': t['query_id'], 'answerable': t['answerable'], 'status': t['status'],
                 'citation_support': [None] * len(t['citations'])}
        if t['answerable']:
            # Câu có đáp án: chấm 0/1/2. Nếu agent từ chối thì đề quy định chấm 0 -> điền sẵn.
            label['answer_grade'] = 0 if t['status'] == 'abstain' else None
        else:
            # Câu thiếu bằng chứng: người chấm xác nhận từ chối đúng (true/false).
            label['abstention_correct'] = None
        label['note'] = ''
        reviews.append(label)
    cases = [{'id': q['id'], 'split': q['split'], 'answerable': q['answerable']} for q in questions]

    out = ROOT / 'results' / 'agent' / f'{a.split}-{a.tag}'
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (('traces.json', traces), ('summary_auto.json', summary), ('cases.json', cases)):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    template = out / 'reviews_template.json'
    template.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'config'}, ensure_ascii=False, indent=2))
    print(f'-> {out}  (copy reviews_template.json -> reviews.json rồi chấm tay)')


if __name__ == '__main__':
    main()
