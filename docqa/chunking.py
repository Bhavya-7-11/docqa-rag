"""Document loading and chunking (pure Python, no heavy imports)."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    source: str     # file name the text came from
    chunk_id: int   # position within that file
    text: str


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping windows, preferring to cut at spaces."""
    if overlap >= size // 2:
        raise ValueError("overlap must be smaller than half of size")
    text = text.strip()
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return pieces


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """Return (file_name, text) for every .md / .txt file in docs_dir."""
    docs = []
    for path in sorted(Path(docs_dir).glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def build_chunks(docs: list[tuple[str, str]], size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for name, text in docs:
        for i, piece in enumerate(chunk_text(text, size, overlap)):
            chunks.append(Chunk(source=name, chunk_id=i, text=piece))
    return chunks
