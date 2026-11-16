"""Chọn context: neighbor expansion — kéo thêm chunk liền trước/sau trong tài liệu."""


def chunk_position(chunk):
    """Thứ tự đọc: (tài liệu, trang, số thứ tự chunk trong trang). ID có dạng ...:pNNN:fixed500o50:007."""
    return chunk['source_id'], chunk['page'], int(chunk['id'].rsplit(':', 1)[1])


class NeighborExpander:
    def __init__(self, chunks, window=1):
        self.ordered = sorted(chunks, key=chunk_position)
        self.pos = {c['id']: i for i, c in enumerate(self.ordered)}
        self.window = window

    def expand(self, hits):
        """Với mỗi hit (theo thứ hạng): [trước..., hit, sau...]; bỏ trùng; không vượt sang tài liệu khác."""
        out, seen = [], set()
        for h in hits:
            i = self.pos[h['id']]
            for j in range(i - self.window, i + self.window + 1):
                if not 0 <= j < len(self.ordered):
                    continue
                c = self.ordered[j]
                if c['source_id'] != h['source_id'] or c['id'] in seen:
                    continue
                seen.add(c['id'])
                out.append(h if j == i else {**c, 'score': h['score'], 'rank': h['rank'],
                                             'strategy': h['strategy'], 'neighbor_of': h['id']})
        return out
