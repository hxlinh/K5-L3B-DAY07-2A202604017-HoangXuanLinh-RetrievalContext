"""Kiểm tra benchmark nâng cao: số câu theo nhóm/split đúng yêu cầu ADVANCED.md, test không dùng trang pilot,
mọi trích dẫn text nằm đúng trang PDF. Bảng/hình phải kiểm bằng mắt với bản gốc (script không thay được).

Chạy: python -X utf8 advanced/benchmark/check_benchmark.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / 'data' / 'raw' / 'Principles-of-Data-Science-WEB.pdf'
PILOT_PAGES = {27, 55, 431}           # đã dùng để chỉnh prompt vision -> không được dùng cho test
MIN_TEST = {'table': 6, 'image': 5, 'chart': 5, 'multihop': 4, 'text': 4, 'unanswerable': 6}
MIN_DEV_ANSWERABLE, MIN_DEV_UNANSWERABLE = 15, 3


def squash(text):
    return re.sub(r'\s+', ' ', text).strip()


def main():
    qs = json.loads((HERE / 'questions.json').read_text(encoding='utf-8'))
    doc = pymupdf.open(PDF)
    errors = []
    if len({q['id'] for q in qs}) != len(qs):
        errors.append('ID bị trùng')
    by = Counter((q['split'], q['group']) for q in qs)
    for group, need in MIN_TEST.items():
        if by[('test', group)] < need:
            errors.append(f'test/{group}: {by[("test", group)]} < {need}')
    dev_ans = sum(n for (s, g), n in by.items() if s == 'dev' and g != 'unanswerable')
    if dev_ans < MIN_DEV_ANSWERABLE or by[('dev', 'unanswerable')] < MIN_DEV_UNANSWERABLE:
        errors.append(f'dev: {dev_ans} có đáp án, {by[("dev", "unanswerable")]} thiếu bằng chứng')
    for q in qs:
        if q['group'] == 'unanswerable':
            if not q.get('absence_check') or q.get('expected_behavior') != 'search_and_abstain':
                errors.append(f"{q['id']}: thiếu absence_check/expected_behavior")
            continue
        pages = {e['page'] for e in q['evidence']}
        if q['split'] == 'test' and pages & PILOT_PAGES:
            errors.append(f"{q['id']}: test dùng trang pilot {pages & PILOT_PAGES}")
        if q['group'] == 'multihop' and len(q['evidence']) < 2:
            errors.append(f"{q['id']}: multi-hop cần >= 2 evidence")
        for e in q['evidence']:
            if 'quote' in e and squash(e['quote']) not in squash(doc[e['page'] - 1].get_text()):
                errors.append(f"{q['id']}: trích dẫn không có ở trang {e['page']}: {e['quote'][:50]}")
    for split in ('dev', 'test'):
        print(split, {g: by[(split, g)] for g in MIN_TEST}, 'tổng', sum(n for (s, _), n in by.items() if s == split))
    print('LỖI:' if errors else 'OK: benchmark hợp lệ.', *errors, sep='\n  ')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
