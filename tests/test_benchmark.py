from evaluation.run_retrieval_benchmark import score_cases
from docqa.chunking import Chunk

class FakeStore:
    chunks = [Chunk("a.md", 0, "The limit is 12 days."), Chunk("b.md", 0, "Other policy.")]
    def search(self, query, k):
        return [(self.chunks[1], 0.9), (self.chunks[0], 0.8)][:k]

def test_evidence_rank_and_mrr():
    cases = [{"id": "A1", "question": "limit?", "expected_source": "a.md", "evidence": "limit is 12 days"},
             {"id": "U1", "question": "unknown", "expected_source": None, "evidence": None}]
    result = score_cases(FakeStore(), cases, [1, 2])
    assert result["metrics"]["1"]["hit_rate"] == 0
    assert result["metrics"]["2"]["hit_rate"] == 1
    assert result["metrics"]["2"]["mrr"] == 0.5
    assert result["unanswerable_questions_not_scored"] == 1

def test_wrong_passage_same_file_is_not_hit():
    class WrongPassage(FakeStore):
        def search(self, query, k):
            return [(Chunk("a.md", 1, "A completely unrelated rule"), 0.9)][:k]
    result = score_cases(WrongPassage(), [{"id": "A", "question": "limit", "expected_source": "a.md", "evidence": "limit is 12 days"}], [1])
    assert result["metrics"]["1"]["hit_rate"] == 0
