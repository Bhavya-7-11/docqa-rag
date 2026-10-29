# DocQA RAG — local LLM edition

Document question answering with Sentence Transformers embeddings, FAISS retrieval, source citations, and **local Ollama Llama 3.2 (3B)** generation. No paid API key needed. Claude is optional.

## Quick start (macOS)

1. Install Ollama from https://ollama.com/download/mac and open the Ollama app.
2. In Terminal: `ollama pull llama3.2:3b`
3. In the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m pytest -q
python -m docqa ingest
python -m docqa ask "Who approves access requests?"
python -m evaluation.run_retrieval_benchmark --k 1 3 5
python -m evaluation.run_eval --with-llm
```

The legacy `run_eval --with-llm` evaluates **11 original questions** using keyword matching and a basic refusal string check. Treat it as a smoke test, not a rigorous semantic answer-quality benchmark. Manually review `evaluation/results.json`. The extended retrieval benchmark evaluates **36 answerable questions** with evidence-level matching; 9 unanswerable questions are excluded from its retrieval score.

## Measured retrieval baseline (user-run, synthetic demonstration corpus)

15 synthetic policy documents, 27 indexed chunks, 36 answerable retrieval queries, 9 unanswerable queries not scored. Hit Rate@1 **0.833** (30/36); Hit Rate@3 **1.000** (36/36); Hit Rate@5 **1.000** (36/36); MRR@3 **0.912**. These are retrieval-only results, not LLM answer accuracy. The corpus is synthetic and limited; results are not evidence of production performance. Re-run the benchmark after any changes.

## Configuration

Edit `.env` to set `DOCQA_PROVIDER=ollama`, `DOCQA_MODEL=llama3.2:3b` (defaults already configured). The local endpoint defaults to `http://localhost:11434/api/chat`; override with `OLLAMA_URL`. For Claude, install `pip install -r requirements-anthropic.txt` and set `DOCQA_PROVIDER=anthropic`, `DOCQA_MODEL` to an available Anthropic model ID, and `ANTHROPIC_API_KEY` in `.env`. Never commit `.env`.

## Architecture

`.md`/`.txt` docs → character chunking with overlap → MiniLM embeddings → FAISS index → top-K retrieval → local Ollama or optional Claude → grounded answer with numbered citations and retrieved-source previews.

## Limitations

The app is a CLI prototype. It does not guarantee that citations are accurate or that the model refuses unsupported questions. Synthetic benchmarks may overestimate performance. Independently evaluate answer correctness and abstention before making claims about them.
