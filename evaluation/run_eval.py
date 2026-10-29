"""Evaluate retrieval (free, offline) and optionally end-to-end answers (uses the LLM API).

  python -m evaluation.run_eval              # retrieval metrics only
  python -m evaluation.run_eval --with-llm   # + answer correctness and refusal checks
"""
import argparse
import json
from pathlib import Path

from docqa.config import INDEX_DIR, TOP_K
from docqa.store import VectorStore

EVAL_PATH = Path(__file__).parent / "eval_set.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("-k", type=int, default=TOP_K)
    args = ap.parse_args()

    store = VectorStore.load(INDEX_DIR)
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    answerable = [c for c in cases if c["expected_source"]]

    hits_at_k, rr = 0, 0.0
    for c in answerable:
        ranked = [ch.source for ch, _ in store.search(c["question"], args.k)]
        if c["expected_source"] in ranked:
            hits_at_k += 1
            rr += 1 / (ranked.index(c["expected_source"]) + 1)
    n = len(answerable)
    report = {"k": args.k, "answerable_questions": n,
              "retrieval_hit_rate": round(hits_at_k / n, 3), "mrr": round(rr / n, 3)}

    if args.with_llm:
        from docqa.qa import answer
        correct = refused_ok = unanswerable = 0
        details = []
        for c in cases:
            out = answer(c["question"], store, k=args.k)
            text = out["answer"].lower()
            if c["expected_source"]:
                ok = all(kw.lower() in text for kw in c["expected_keywords"])
                correct += ok
            else:
                unanswerable += 1
                ok = "don't know" in text
                refused_ok += ok
            details.append({"question": c["question"], "pass": ok, "answer": out["answer"]})
        report["answer_accuracy"] = round(correct / n, 3)
        report["correct_refusals"] = f"{refused_ok}/{unanswerable}"
        Path(__file__).parent.joinpath("results.json").write_text(json.dumps(details, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
