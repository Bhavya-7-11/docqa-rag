"""Retrieval-augmented answering with source citations."""
import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from .config import LLM_MODEL, TOP_K, LLM_PROVIDER
from .store import VectorStore

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM = f"""You answer questions using ONLY the numbered context passages provided.
Rules:
- Cite the passages you used like [1] or [2][3] right after the claim.
- If the passages do not contain the answer, reply exactly: {NO_ANSWER}
- Do not use outside knowledge. Be concise."""


def build_prompt(question: str, hits) -> str:
    blocks = [f"[{i}] (source: {c.source})\n{c.text}" for i, (c, _) in enumerate(hits, 1)]
    return "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def answer(question: str, store: VectorStore, k: int = TOP_K, model: str = LLM_MODEL) -> dict:
    hits = store.search(question, k)
    prompt = build_prompt(question, hits)
    if LLM_PROVIDER == "ollama":
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        payload = json.dumps({"model": model, "stream": False, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ], "options": {"temperature": 0}}).encode("utf-8")
        request = Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=180) as response:
                text = json.load(response)["message"]["content"].strip()
        except (URLError, HTTPError) as exc:
            raise RuntimeError("Ollama request failed. Start Ollama and run: ollama pull llama3.2:3b") from exc
    elif LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic
        resp = Anthropic().messages.create(model=model, max_tokens=512, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(block.text for block in resp.content if block.type == "text")
    else:
        raise ValueError(f"Unsupported DOCQA_PROVIDER: {LLM_PROVIDER}")
    sources = [
        {"n": i, "source": c.source, "chunk_id": c.chunk_id, "score": round(s, 3), "text": c.text}
        for i, (c, s) in enumerate(hits, 1)
    ]
    return {"question": question, "answer": text, "sources": sources}
