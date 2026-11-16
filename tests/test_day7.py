"""Kiểm tra hành vi code Day 7 (src/) bằng dữ liệu giả; không đo chất lượng model hay retrieval thật."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agent import ABSTAIN_TEXT, run_agent
from context import NeighborExpander
from eval_retrieval import is_hit
from retrievers import BM25Retriever, HybridRRF


def chunk(cid, text, page=1, source='doc'):
    return {'id': f'{source}:p{page}:fixed:{cid:03d}', 'text': text, 'source_id': source,
            'source': f'{source}.pdf', 'page': page, 'metadata': {'chapter': 2 if source == 'doc' else 8}}


CHUNKS = [chunk(0, 'Stratified sampling splits the population into subgroups.'),
          chunk(1, 'Each stratum is sampled in proportion to its size.'),
          chunk(2, 'Cluster sampling picks whole groups.'),
          chunk(0, 'Pseudonymization replaces values with artificial codes.', page=5, source='other')]


class Scripted:
    """Controller giả: trả lần lượt các action đã định sẵn."""
    def __init__(self, actions):
        self.actions, self.seen = iter(actions), []

    def decide(self, question, state):
        self.seen.append(state)
        return next(self.actions)


class Fixed:
    name = 'fixed'

    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k))
        return [{**h, 'score': 1.0, 'rank': i + 1, 'strategy': self.name} for i, h in enumerate(self.hits[:top_k])]


class AgentTests(unittest.TestCase):
    def test_answer_with_citation_in_context(self):
        cid = CHUNKS[0]['id']
        c = Scripted([{'action': 'search', 'query': 'stratified'},
                      {'action': 'answer', 'answer': f'Subgroups [{cid}]', 'citations': [cid]}])
        r = run_agent('q', c, Fixed([CHUNKS[0]]), filters={'chapter': 2})
        self.assertEqual((r['status'], r['citations'], r['stop_reason']), ('answered', [cid], 'answer'))
        self.assertEqual(r['events'][0]['filter'], {'chapter': 2})

    def test_citation_outside_context_is_rejected(self):
        c = Scripted([{'action': 'search', 'query': 'x'},
                      {'action': 'answer', 'answer': 'X [made-up]', 'citations': ['made-up']}])
        r = run_agent('q', c, Fixed([CHUNKS[0]]))
        self.assertEqual((r['status'], r['answer'], r['stop_reason']), ('abstain', ABSTAIN_TEXT, 'invalid_or_missing_citations'))
        self.assertEqual(r['events'][-1]['bad_citations'], ['made-up'])

    def test_model_abstain_records_reason(self):
        c = Scripted([{'action': 'search', 'query': 'x'}, {'action': 'abstain', 'reason': 'not in book'}])
        r = run_agent('q', c, Fixed([CHUNKS[2]]))
        self.assertEqual((r['status'], r['stop_reason']), ('abstain', 'model_abstain'))
        self.assertTrue(r['events'][-1]['searched_before_abstain'])

    def test_search_limit(self):
        c = Scripted([{'action': 'search', 'query': 'a'}, {'action': 'search', 'query': 'b'}])
        retriever = Fixed([CHUNKS[0]])
        r = run_agent('q', c, retriever, max_searches=1)
        self.assertEqual((r['status'], r['stop_reason'], len(retriever.calls)), ('abstain', 'search_limit', 1))

    def test_budget_grows_per_search(self):
        c = Scripted([{'action': 'search', 'query': 'a'}, {'action': 'search', 'query': 'b'},
                      {'action': 'abstain', 'reason': 'r'}])
        hits = iter([[CHUNKS[0]], [CHUNKS[1]]])

        class Seq(Fixed):
            def search(self, query, *, filters, top_k):
                return [{**h, 'score': 1.0, 'rank': 1, 'strategy': 'seq'} for h in next(hits)]
        r = run_agent('q', c, Seq([]), budget_per_search=120)
        self.assertEqual(r['events'][0]['context_ids'], [CHUNKS[0]['id']])
        self.assertEqual(r['events'][1]['context_ids'], [CHUNKS[0]['id'], CHUNKS[1]['id']])

    def test_provider_errors_propagate(self):
        class Broken:
            def decide(self, question, state):
                raise RuntimeError('OpenAI HTTP 500')
        with self.assertRaises(RuntimeError):
            run_agent('q', Broken(), Fixed([CHUNKS[0]]))


class ContextAndMetricTests(unittest.TestCase):
    def test_neighbors_stay_inside_document_and_dedupe(self):
        ex = NeighborExpander(CHUNKS, window=1)
        hits = [{**CHUNKS[1], 'score': 1, 'rank': 1, 'strategy': 's'}, {**CHUNKS[0], 'score': 1, 'rank': 2, 'strategy': 's'}]
        ids = [c['id'] for c in ex.expand(hits)]
        self.assertEqual(ids, [CHUNKS[0]['id'], CHUNKS[1]['id'], CHUNKS[2]['id']])
        self.assertNotIn(CHUNKS[3]['id'], ids)

    def test_hit_any_and_all(self):
        gold = [{'source_id': 'doc', 'page': 1, 'evidence': 'splits the  population'},
                {'source_id': 'doc', 'page': 1, 'evidence': 'in proportion to its size'}]
        self.assertTrue(is_hit({'gold': gold}, [CHUNKS[0]]))                       # any, khoảng trắng chuẩn hóa
        self.assertFalse(is_hit({'gold': gold, 'evidence_mode': 'all'}, [CHUNKS[0]]))
        self.assertTrue(is_hit({'gold': gold, 'evidence_mode': 'all'}, CHUNKS[:2]))
        wrong_page = [{**gold[0], 'page': 2}]
        self.assertFalse(is_hit({'gold': wrong_page}, CHUNKS))


class RetrievalTests(unittest.TestCase):
    def test_bm25_ranks_rare_term_and_applies_filter(self):
        bm25 = BM25Retriever(CHUNKS)
        self.assertEqual(bm25.search('pseudonymization codes', top_k=1)[0]['id'], CHUNKS[3]['id'])
        self.assertEqual(bm25.search('pseudonymization codes', filters={'chapter': 2}, top_k=3), [])
        with self.assertRaises(ValueError):
            bm25.search('x', filters={'year': 2025})

    def test_rrf_rewards_agreement(self):
        a = Fixed([CHUNKS[0], CHUNKS[1]]); a.name = 'a'
        b = Fixed([CHUNKS[1], CHUNKS[2]]); b.name = 'b'
        top = HybridRRF([a, b], k=60, candidates=2).search('q', top_k=1)[0]
        self.assertEqual(top['id'], CHUNKS[1]['id'])
        self.assertEqual(top['component_ranks'], {'a': 2, 'b': 1})


if __name__ == '__main__':
    unittest.main()
