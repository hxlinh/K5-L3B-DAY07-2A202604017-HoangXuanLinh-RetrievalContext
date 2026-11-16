"""Export một index từ pgvector Day 6 sang schema chunk Day 7 (JSON array, UTF-8)."""
import argparse
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='A', help='Index Day 6 (A đã được chọn ở Day 6)')
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'day7' / 'chunks.json')
    args = parser.parse_args()

    load_dotenv(ROOT / 'starter' / '.env')
    with psycopg.connect(os.environ['DATABASE_URL'], connect_timeout=5) as conn:
        rows = conn.execute(
            """SELECT chunk_id, document_id, source, page, text, metadata, model
               FROM chunks WHERE strategy = %s ORDER BY chunk_id""",
            (args.strategy,)).fetchall()
    if not rows:
        raise SystemExit(f'Không có chunk nào cho strategy {args.strategy}; DB đã bật đúng project chưa?')

    chunks = [{'id': cid, 'source_id': doc, 'source': src, 'page': page, 'text': text,
               'metadata': {**meta, 'index': args.strategy, 'embedding_model': model}}
              for cid, doc, src, page, text, meta, model in rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(chunks)} chunks -> {args.output}')
    print('Embedding models:', sorted({c['metadata']['embedding_model'] for c in chunks}))


if __name__ == '__main__':
    main()
