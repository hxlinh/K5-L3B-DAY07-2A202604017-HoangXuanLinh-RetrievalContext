"""Bước A1: khảo sát PDF (không gọi API) — trang nào có ảnh, biểu đồ vẽ vector, bảng.

Kết quả dùng để quyết định trang nào cần gửi cho vision model và ước lượng chi phí.
Chạy: python -X utf8 advanced/profile_pdf.py
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parent
CAPTION = re.compile(r'\b(Figure|Table)\s+(\d+\.\d+)')
# Chú thích thật nằm ở đầu dòng ("Table 4.1 ..."); "as shown in Figure 9.3" chỉ là nhắc tới, hình nằm trang khác.
CAPTION_LINE = re.compile(r'^(Figure|Table)\s+(\d+\.\d+)\b', re.M)


def chapter_map(doc):
    """Trang (1-based) -> tên chương cấp 1, lấy từ mục lục của chính PDF."""
    starts = [(page, title) for level, title, page in doc.get_toc() if level == 1 and page > 0]
    starts.sort()
    result = {}
    for i, (start, title) in enumerate(starts):
        end = starts[i + 1][0] - 1 if i + 1 < len(starts) else doc.page_count
        for p in range(start, end + 1):
            result[p] = title
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pdf', type=Path, default=ROOT / 'data' / 'raw' / 'Principles-of-Data-Science-WEB.pdf')
    p.add_argument('--output', type=Path, default=ROOT / 'data' / 'page_profile.jsonl')
    a = p.parse_args()

    doc = fitz.open(a.pdf)
    chapters = chapter_map(doc)
    rows = []
    for page in doc:
        n = page.number + 1
        text = page.get_text()
        images = page.get_images(full=True)
        # Ảnh quá nhỏ (logo, icon) không mang thông tin -> không tính.
        big_images = [img for img in images
                      if any(r.width * r.height > 0.03 * page.rect.width * page.rect.height
                             for r in page.get_image_rects(img[0]))]
        drawings = len(page.get_drawings())          # biểu đồ/sơ đồ vẽ bằng vector, không phải ảnh
        mentioned = sorted({f'{kind} {num}' for kind, num in CAPTION.findall(text)})
        captions = sorted({f'{kind} {num}' for kind, num in CAPTION_LINE.findall(text)})
        has_table = any(c.startswith('Table') for c in captions)
        has_figure = any(c.startswith('Figure') for c in captions)
        rows.append({
            'page': n, 'chapter': chapters.get(n, 'front matter'), 'chars': len(text.strip()),
            'images': len(images), 'big_images': len(big_images), 'drawings': drawings,
            'captions': captions, 'mentioned': mentioned,
            'has_table_caption': has_table, 'has_figure_caption': has_figure,
            # Ứng viên gửi vision: có ảnh lớn hoặc có chú thích Table/Figure ở đầu dòng.
            # Không dùng số nét vẽ: kiểm tra trang 115, 131 cho thấy nét vẽ chủ yếu là công thức
            # toán (gạch phân số) và khung Example, không phải biểu đồ.
            'vision_candidate': bool(big_images) or has_table or has_figure,
        })

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    cand = [r for r in rows if r['vision_candidate']]
    print(f'PDF: {a.pdf.name} | {doc.page_count} trang | trang có chữ (>200 ký tự): '
          f'{sum(r["chars"] > 200 for r in rows)}')
    print(f'Trang có ảnh lớn: {sum(r["big_images"] > 0 for r in rows)} | '
          f'có >=30 nét vẽ (chỉ tham khảo): {sum(r["drawings"] >= 30 for r in rows)} | '
          f'có chú thích Table: {sum(r["has_table_caption"] for r in rows)} | '
          f'Figure: {sum(r["has_figure_caption"] for r in rows)}')
    print(f'=> Ứng viên vision: {len(cand)} trang')
    print('\nTheo chương (ứng viên vision / tổng trang):')
    total = Counter(r['chapter'] for r in rows)
    vis = Counter(r['chapter'] for r in cand)
    for ch in total:
        print(f'  {vis[ch]:>3} / {total[ch]:<3} {ch}')
    print(f'\n-> {a.output}')


if __name__ == '__main__':
    main()
