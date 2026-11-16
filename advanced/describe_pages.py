"""Bước A2: render trang thành ảnh, gửi vision model (gpt-5.4-mini) để trích bảng/biểu đồ/ảnh.

- Chỉ gửi các trang vision_candidate trong page_profile.jsonl (hoặc --pages chỉ định).
- Chặn ngân sách: tổng chi phí trong file kết quả (cả các lần chạy trước) không vượt --max-usd.
- Chạy tiếp được: trang đã có trong file kết quả sẽ bỏ qua.
- Ảnh render được giữ lại (advanced/data/page_images/) để agent xem lại lúc trả lời.

Chạy thử 3 trang: python -X utf8 advanced/describe_pages.py --pages 27 55 431 --max-usd 0.05
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import pymupdf as fitz
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(HERE))
import budget
from llm import OpenAIChat

PROMPT = """You extract visual content from one page of the textbook "Principles of Data Science" (OpenStax) for a search index.
The page image is attached, followed by the text extracted from the same page.

List every informative visual element on the page: tables, charts/graphs, diagrams, and images or screenshots that carry information (for example a screenshot of a data table or of program output).
Ignore logos, decorative icons, photos without data, and ordinary paragraphs of text.

Return JSON: {"elements": [ {
  "type": "table" | "chart" | "diagram" | "image",
  "label": "the printed label such as \\"Figure 9.3\\" or \\"Table 4.1\\", or null",
  "title": "caption or title as printed",
  "description": "2-4 sentences: what it shows and the main takeaway",
  "data": "tables: a complete markdown table with every row, column and unit; charts: chart type, axis titles and units, legend, and every readable data value or category; diagrams/images: all visible labels and values"
} ]}
Copy numbers and labels exactly as printed. If a value is not readable, write "unreadable"; never guess.
For bar charts and histograms: name each bar by the label printed directly under its center (a bar centered over the tick "7" is "7", not "7-8"). If the value axis has no numbers but has gridlines, give each bar's height in gridline units, counting carefully from the baseline (for example "7: 6 gridlines").
For line and scatter charts: give the axis ranges and the coordinates of labeled or clearly readable points.
If there is no informative visual element, return {"elements": []}."""


def render(doc, page_no, dpi, out_dir):
    path = out_dir / f'p{page_no:03d}.png'
    if not path.exists():
        doc[page_no - 1].get_pixmap(dpi=dpi).save(path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pdf', type=Path, default=HERE / 'data' / 'raw' / 'Principles-of-Data-Science-WEB.pdf')
    p.add_argument('--profile', type=Path, default=HERE / 'data' / 'page_profile.jsonl')
    p.add_argument('--output', type=Path, default=HERE / 'data' / 'visual_descriptions.jsonl')
    p.add_argument('--pages', type=int, nargs='*', help='chỉ xử lý các trang này (mặc định: mọi vision_candidate)')
    p.add_argument('--dpi', type=int, default=110, help='110 dpi ~ 935x1210 px cho trang A4/Letter')
    p.add_argument('--max-usd', type=float, required=True,
                   help='dừng khi TỔNG sổ cái chi phí (data/api_costs.jsonl) chạm mức này')
    p.add_argument('--redo', action='store_true', help='xóa kết quả cũ của các trang --pages rồi chạy lại')
    a = p.parse_args()

    load_dotenv(REPO / 'starter' / '.env')
    llm = OpenAIChat(timeout=120)
    doc = fitz.open(a.pdf)
    img_dir = HERE / 'data' / 'page_images'
    img_dir.mkdir(parents=True, exist_ok=True)

    profile = {r['page']: r for r in map(json.loads, a.profile.read_text(encoding='utf-8').splitlines())}
    pages = a.pages or [n for n, r in profile.items() if r['vision_candidate']]
    rows = [json.loads(l) for l in a.output.read_text(encoding='utf-8').splitlines()] if a.output.exists() else []
    if a.redo and a.pages:
        rows = [r for r in rows if r['page'] not in set(a.pages)]
        a.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    done = {r['page'] for r in rows}
    todo = [n for n in pages if n not in done]
    print(f'{len(pages)} trang được chọn, {len(done & set(pages))} đã xong, còn {len(todo)}. '
          f'Sổ cái đã tiêu: ${budget.spent():.4f}')

    with a.output.open('a', encoding='utf-8', newline='\n') as f:
        for n in todo:
            try:
                budget.check(a.max_usd)
            except SystemExit as stop:
                print(stop)
                break
            img = render(doc, n, a.dpi, img_dir)
            b64 = base64.b64encode(img.read_bytes()).decode()
            text = doc[n - 1].get_text()[:3000]
            t0 = time.perf_counter()
            resp = llm.complete([{'role': 'user', 'content': [
                {'type': 'text', 'text': PROMPT},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}', 'detail': 'high'}},
                {'type': 'text', 'text': f'EXTRACTED PAGE TEXT (page {n}):\n{text}'}]}],
                response_format={'type': 'json_object'})
            usage = resp['usage']
            cost = budget.record('describe_pages', usage, page=n)
            try:
                elements = json.loads(resp['message']['content'])['elements']
                error = None
            except (json.JSONDecodeError, KeyError, TypeError) as e:     # ghi lỗi, không bịa nội dung
                elements, error = [], f'{type(e).__name__}: {e}'
            row = {'page': n, 'chapter': profile[n]['chapter'], 'image': str(img.relative_to(REPO)).replace('\\', '/'),
                   'dpi': a.dpi, 'elements': elements, 'error': error, 'model': resp['model'], 'usage': usage,
                   'cost_usd': round(cost, 6), 'latency_s': round(time.perf_counter() - t0, 1)}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            kinds = ','.join(e.get('type', '?') for e in elements) or '-'
            print(f"p{n:03d} {len(elements)} phần tử [{kinds}] tok={usage.get('prompt_tokens')}+{usage.get('completion_tokens')} "
                  f"${cost:.4f} | sổ cái ${budget.spent():.4f} {error or ''}")
    print(f'-> {a.output}')


if __name__ == '__main__':
    main()
