"""Vector store: sentence-transformers embeddings + FAISS inner-product index."""
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .chunking import Chunk


@lru_cache(maxsize=2)
def _embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _embed(texts: list[str], model_name: str) -> np.ndarray:
    # Normalised vectors => inner product == cosine similarity.
    vecs = _embedder(model_name).encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    )
    return vecs.astype("float32")


class VectorStore:
    def __init__(self, index, chunks: list[Chunk], model_name: str):
        self.index = index
        self.chunks = chunks
        self.model_name = model_name

    @classmethod
    def build(cls, chunks: list[Chunk], model_name: str) -> "VectorStore":
        import faiss
        vecs = _embed([c.text for c in chunks], model_name)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(index, chunks, model_name)

    def save(self, directory: Path) -> None:
        import faiss
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        meta = {"model": self.model_name, "chunks": [asdict(c) for c in self.chunks]}
        (directory / "chunks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        import faiss
        meta_path = directory / "chunks.json"
        if not meta_path.exists():
            raise FileNotFoundError("No index found. Run: python -m docqa ingest")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(directory / "faiss.index"))
        return cls(index, [Chunk(**c) for c in meta["chunks"]], meta["model"])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = _embed([query], self.model_name)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
