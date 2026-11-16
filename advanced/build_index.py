"""Bước A4: xây index cho toàn bộ sách (569 trang), miễn phí, chạy local trên CPU.

Ba loại vector, cùng model google/embeddinggemma-2 (768 chiều, bật bộ mã hóa ảnh):
  text        : chunk text mọi trang (500 ký tự, chồng lấn 50 — giống index A Day 6), prompt Document
  image       : 1 vector / trang có hình-bảng (vision_candidate), chỉ từ ảnh trang
  interleaved : 1 vector / trang có hình-bảng, text trang + ảnh trang trong cùng một input

Chạy được tiếp nếu bị ngắt (lưu sau mỗi 10 trang). Kết quả: advanced/data/index/
Chạy: python -X utf8 advanced/build_index.py --kinds text image interleaved
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pymupdf
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUT = DATA / 'index'
PDF = DATA / 'raw' / 'Principles-of-Data-Science-WEB.pdf'
FOOTER = re.compile(r'Access for free at openstax\.org')
SIZE, OVERLAP = 500, 50
INTERLEAVED_TEXT_CHARS = 2000        # text trang đưa kèm ảnh (giới hạn để không chiếm hết context 8192 token)


def page_text(page):
    return re.sub(r'\s+', ' ', FOOTER.sub(' ', page.get_text())).strip()


def chunk_pages(doc, profile):
    chunks = []
    for page in doc:
        n = page.number + 1
        text = page_text(page)
        if len(text) < 50:
            continue
        start, i = 0, 0
        while start < len(text):
            chunks.append({'id': f'pods:p{n:03d}:c{i:02d}', 'page': n, 'chapter': profile[n]['chapter'],
                           'text': text[start:start + SIZE], 'char_start': start})
            if start + SIZE >= len(text):
                break
            start += SIZE - OVERLAP
            i += 1
    return chunks


def load_partial(name):
    meta, vec = OUT / f'{name}_pages.json', OUT / f'{name}_vectors.npy'
    if meta.exists() and vec.exists():
        return json.loads(meta.read_text(encoding='utf-8')), list(np.load(vec))
    return [], []


def save(name, pages, vectors):
    (OUT / f'{name}_pages.json').write_text(json.dumps(pages), encoding='utf-8')
    np.save(OUT / f'{name}_vectors.npy', np.asarray(vectors, dtype=np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kinds', nargs='+', default=['text', 'image', 'interleaved'],
                   choices=['text', 'image', 'interleaved'])
    p.add_argument('--dpi', type=int, default=110)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (DATA / 'page_images').mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(PDF)
    profile = {r['page']: r for r in map(json.loads, (DATA / 'page_profile.jsonl').read_text(encoding='utf-8').splitlines())}
    t0 = time.perf_counter()
    model = SentenceTransformer('google/embeddinggemma-2', config_kwargs={'audio_config': None},
                                model_kwargs={'dtype': torch.float32})
    print(f'model loaded in {time.perf_counter() - t0:.0f}s')
    stats = {}

    if 'text' in a.kinds:
        chunks = chunk_pages(doc, profile)
        t0 = time.perf_counter()
        vecs = model.encode([c['text'] for c in chunks], prompt_name='Document', normalize_embeddings=True,
                            batch_size=16, show_progress_bar=True)
        (OUT / 'text_chunks.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in chunks),
                                               encoding='utf-8')
        np.save(OUT / 'text_vectors.npy', vecs.astype(np.float32))
        stats['text'] = {'chunks': len(chunks), 'pages': len({c['page'] for c in chunks}),
                         'seconds': round(time.perf_counter() - t0)}
        print('text:', stats['text'])

    targets = [n for n, r in profile.items() if r['vision_candidate']]
    for kind in [k for k in ('image', 'interleaved') if k in a.kinds]:
        pages, vectors = load_partial(kind)
        todo = [n for n in targets if n not in set(pages)]
        print(f'{kind}: {len(pages)} pages done, {len(todo)} to go')
        t0 = time.perf_counter()
        for i, n in enumerate(todo, 1):
            img = DATA / 'page_images' / f'p{n:03d}.png'
            if not img.exists():
                doc[n - 1].get_pixmap(dpi=a.dpi).save(img)
            if kind == 'image':
                item = {'text': '<|image|>', 'image': [str(img)]}
            else:
                item = {'text': 'title: none | text: ' + page_text(doc[n - 1])[:INTERLEAVED_TEXT_CHARS] + ' <|image|>',
                        'image': [str(img)]}
            vectors.append(model.encode([item], normalize_embeddings=True, batch_size=1)[0])
            pages.append(n)
            if i % 10 == 0 or i == len(todo):
                save(kind, pages, vectors)
                rate = (time.perf_counter() - t0) / i
                print(f'  {kind} {len(pages)}/{len(targets)}  {rate:.1f}s/page  ETA {rate * (len(todo) - i) / 60:.0f} min')
        stats[kind] = {'pages': len(pages), 'seconds_this_run': round(time.perf_counter() - t0)}

    meta = OUT / 'index_meta.json'
    old = json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else {}
    old.update({'model': 'google/embeddinggemma-2', 'dim': 768, 'dtype': 'float32', 'device': 'cpu',
                'text_chunking': f'{SIZE} chars, overlap {OVERLAP}, per page, footer removed',
                'document_prompt': 'Document (title: none | text: ...)', 'image_dpi': a.dpi,
                'image_scope': 'vision_candidate pages from page_profile.jsonl',
                'interleaved_text_chars': INTERLEAVED_TEXT_CHARS, **{f'stats_{k}': v for k, v in stats.items()}})
    meta.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding='utf-8')
    print('->', OUT)


if __name__ == '__main__':
    main()
