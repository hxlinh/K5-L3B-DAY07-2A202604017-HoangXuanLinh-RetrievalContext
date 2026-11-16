"""Sổ cái chi phí API cho toàn bộ phần nâng cao (ngân sách tối đa ~3 USD).

Mọi lần gọi API đều ghi một dòng vào data/api_costs.jsonl; trần được kiểm tra trên tổng sổ cái,
nên chạy lại hay xóa file kết quả cũng không làm "mất" tiền đã tiêu.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / 'data' / 'api_costs.jsonl'
TOTAL_BUDGET_USD = 3.0


def cost_of(usage):
    p_in, p_out = float(os.environ['PRICE_INPUT_PER_1M']), float(os.environ['PRICE_OUTPUT_PER_1M'])
    return (usage.get('prompt_tokens', 0) * p_in + usage.get('completion_tokens', 0) * p_out) / 1e6


def spent():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)['cost_usd'] for line in LEDGER.read_text(encoding='utf-8').splitlines())


def record(step, usage, **info):
    cost = cost_of(usage)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps({'at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'step': step,
                            'prompt_tokens': usage.get('prompt_tokens', 0),
                            'completion_tokens': usage.get('completion_tokens', 0),
                            'cost_usd': round(cost, 6), **info}, ensure_ascii=False) + '\n')
    return cost


def check(cap):
    """Dừng nếu tổng đã tiêu chạm trần của bước này hoặc ngân sách chung."""
    total = spent()
    limit = min(cap, TOTAL_BUDGET_USD)
    if total >= limit:
        raise SystemExit(f'DỪNG: đã tiêu ${total:.4f} >= trần ${limit:.2f}.')
    return total
