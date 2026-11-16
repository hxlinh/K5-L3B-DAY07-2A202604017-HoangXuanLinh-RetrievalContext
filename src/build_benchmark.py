"""Gộp benchmark Day 7 = câu Day 6 (+ sửa nhãn, ghi rõ lý do) + câu thiếu bằng chứng."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from convert_day6 import convert_question  # tái dùng phần đổi tên trường của starter


def squash(text):
    """Cùng định nghĩa hit với Day 6: gộp mọi khoảng trắng thành một dấu cách."""
    return re.sub(r'\s+', ' ', text).strip()


# Sửa nhãn TRƯỚC mọi thí nghiệm Day 7. Lý do: evidence Day 6 vắt qua ranh giới chunk
# của index A (500 ký tự, overlap 50), nên không chunk nào chứa trọn -> không bao giờ hit.
# Quyết định dựa trên ranh giới chunk và nội dung câu hỏi, không dựa trên kết quả retrieval.
FIXES = {
    'q02': {'gold': [{'source_id': 'pods-ch02', 'page': 68, 'evidence':
            'Observational data, often used in qualitative research, is collected by '
            'systematically observing and recording behavior'}]},
    'q22': {'gold': [{'source_id': 'pods-ch08', 'page': 411, 'evidence':
            'if the data utilized in graphs or reports are not publicly available, '
            'permission must be obtained from the source'}],
            'expected_answer': 'When the data is not publicly available: permission must be '
                               'obtained from the source.'},
    'q09': {'evidence_mode': 'all', 'gold': [
            {'source_id': 'pods-ch02', 'page': 87, 'evidence':
             'One approach is to delete the outliers from the dataset, but this can also lead '
             'to a loss of valuable information'},
            {'source_id': 'pods-ch02', 'page': 87, 'evidence':
             'Another option is to deal with the outliers as an individual group and analyze '
             'them separately from the rest of the data.'}]},
    'q16': {'evidence_mode': 'all', 'gold': [
            {'source_id': 'pods-ch08', 'page': 399, 'evidence':
             'an investigation in 2016 found that the algorithm was biased against Black defendants.'},
            {'source_id': 'pods-ch08', 'page': 399, 'evidence':
             'the algorithm incorrectly flagged Black defendants as high risk at nearly twice '
             'the rate as White defendants'}]},
    'q13': {'gold': [
            {'source_id': 'pods-ch08', 'page': 390, 'evidence':
             'Pseudonymization is the act of replacing sensitive information in a dataset with '
             'artificial identifiers or codes while still maintaining its usefulness for analysis.'},
            {'source_id': 'pods-ch08', 'page': 401, 'evidence':
             'Pseudonymization of data involves masking, encoding, or encrypting the values of '
             'specific features of the data.'}]},
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    chunks = load(ROOT / 'data' / 'day7' / 'chunks.json')
    day6 = load(ROOT / 'data' / 'day6' / 'questions.json')
    unanswerable = load(ROOT / 'data' / 'day7' / 'unanswerable.json')

    questions = []
    for raw in day6:
        q = convert_question(raw)
        q.update(type=raw['type'], language=raw['language'], answerable=True,
                 expected_behavior='search_and_answer', evidence_mode='any')
        if q['id'] in FIXES:
            q.update(FIXES[q['id']])
            q['label_fix'] = 'xem FIXES: rút gọn/tách phần (ranh giới chunk) hoặc thêm evidence thay thế'
        questions.append(q)

    # Kiểm tra: mọi phần evidence phải nằm trọn trong ÍT NHẤT một chunk đúng nguồn + trang.
    for q in questions:
        for g in q['gold']:
            found = [c['id'] for c in chunks if c['source_id'] == g['source_id']
                     and c['page'] == g['page'] and squash(g['evidence']) in squash(c['text'])]
            if not found:
                raise SystemExit(f"{q['id']}: evidence không nằm trọn trong chunk nào: {g['evidence'][:60]}")

    questions += unanswerable
    out = ROOT / 'data' / 'day7' / 'questions.json'
    out.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    for split in ('dev', 'test'):
        rows = [q for q in questions if q['split'] == split]
        ans = sum(q['answerable'] for q in rows)
        print(f'{split}: {ans} có đáp án + {len(rows) - ans} thiếu bằng chứng = {len(rows)}')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
