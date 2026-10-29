"""Evidence-level retrieval benchmark. Requires only local embeddings and FAISS.

Run: python -m evaluation.run_retrieval_benchmark --k 1 3 5
"""
import argparse
import json
from pathlib import Path

from docqa.config import INDEX_DIR
from docqa.store import VectorStore

HERE = Path(__file__).parent

def score_cases(store, cases, ks):
    eligible = [c for c in cases if c["evidence"]]
    if not eligible:
        raise ValueError("No answerable questions in evaluation set")
    if max(ks) > len(store.chunks):
        raise ValueError(f"Requested k={max(ks)} but index contains only {len(store.chunks)} chunks. Re-run ingestion.")
    rows = []
    for case in eligible:
        results = store.search(case["question"], max(ks))
        ranks = [i + 1 for i, (chunk, _) in enumerate(results)
                 if chunk.source == case["expected_source"]
                 and case["evidence"].casefold() in chunk.text.casefold()]
        rank = min(ranks) if ranks else None
        rows.append({"id": case["id"], "question": case["question"],
                     "expected_source": case["expected_source"],
                     "evidence": case["evidence"], "first_evidence_rank": rank,
                     "retrieved": [{"source": chunk.source, "chunk_id": chunk.chunk_id,
                                    "score": round(score, 4)} for chunk, score in results]})
    n = len(rows)
    metrics = {}
    for k in ks:
        hits = sum(r["first_evidence_rank"] is not None and r["first_evidence_rank"] <= k for r in rows)
        reciprocal = sum(1 / r["first_evidence_rank"] for r in rows
                         if r["first_evidence_rank"] is not None and r["first_evidence_rank"] <= k)
        metrics[str(k)] = {"hit_rate": round(hits / n, 3), "mrr": round(reciprocal / n, 3),
                           "hits": hits, "total": n}
    return {"evaluation_type": "evidence-level retrieval", "answerable_questions": n,
            "unanswerable_questions_not_scored": len(cases) - n,
            "indexed_chunks": len(store.chunks), "metrics": metrics, "details": rows}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--dataset", type=Path, default=HERE / "eval_set_extended.json")
    parser.add_argument("--output", type=Path, default=HERE / "retrieval_benchmark_results.json")
    args = parser.parse_args()
    if not args.k or any(k < 1 for k in args.k):
        parser.error("k values must be positive integers")
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    report = score_cases(VectorStore.load(INDEX_DIR), cases, sorted(set(args.k)))
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    print(f"Saved question-level details to {args.output}")

if __name__ == "__main__":
    main()
