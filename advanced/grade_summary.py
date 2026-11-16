"""Tổng hợp nhãn chấm tay đạt/không đạt (reviews.json) của một hoặc nhiều lần chạy agent.

reviews.json: danh sách {query_id, group, status, pass: true/false, note}. Chấm theo ADVANCED.md:
câu có đáp án đạt khi đúng, đủ và citation hỗ trợ; câu thiếu bằng chứng đạt khi từ chối hoặc nói rõ không đủ bằng chứng.
Chạy: python -X utf8 advanced/grade_summary.py dev-hinter-k5 dev-inter-k5
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUPS = ['table', 'image', 'chart', 'multihop', 'text', 'unanswerable']
THRESHOLD = 0.9


def summarize(tag):
    path = HERE / 'results' / 'agent' / tag / 'reviews.json'
    rows = json.loads(path.read_text(encoding='utf-8'))
    missing = [r['query_id'] for r in rows if r['pass'] is None]
    if missing:
        sys.exit(f'{tag}: còn câu chưa chấm: {missing}')
    cost = json.loads((path.parent / 'summary_auto.json').read_text(encoding='utf-8'))['cost_usd']
    by_group = defaultdict(lambda: [0, 0])
    for r in rows:
        by_group[r['group']][0] += r['pass']
        by_group[r['group']][1] += 1
    passed = sum(r['pass'] for r in rows)
    result = {'tag': tag, 'passed': passed, 'total': len(rows), 'rate': round(passed / len(rows), 3),
              'meets_90': passed / len(rows) >= THRESHOLD, 'cost_usd': cost,
              'by_group': {g: f'{by_group[g][0]}/{by_group[g][1]}' for g in GROUPS if g in by_group},
              'failed': [r['query_id'] for r in rows if not r['pass']]}
    (path.parent / 'grade_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n',
                                                     encoding='utf-8')
    return result


def main():
    tags = sys.argv[1:] or sys.exit('Cho ít nhất một tag, ví dụ: dev-hinter-k5')
    results = [summarize(t) for t in tags]
    print(f"{'run':18}{'pass':>8}{'rate':>7}{'≥90%':>6}{'$':>8}   " + '  '.join(f'{g[:6]:>6}' for g in GROUPS))
    for r in results:
        print(f"{r['tag']:18}{r['passed']:>4}/{r['total']:<3}{r['rate']:>7.0%}{'yes' if r['meets_90'] else 'no':>6}"
              f"{r['cost_usd']:>8.3f}   " + '  '.join(f"{r['by_group'].get(g, '-'):>6}" for g in GROUPS))
        print(f"{'':18}failed: {', '.join(r['failed']) or '-'}")


if __name__ == '__main__':
    main()
