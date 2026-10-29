import pytest

from docqa.chunking import build_chunks, chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # Consecutive chunks share some text because of the overlap.
    assert chunks[0].split()[-1] in chunks[1]


def test_all_content_is_covered():
    text = " ".join(f"w{i}" for i in range(200))
    joined = " ".join(chunk_text(text, size=120, overlap=20))
    assert all(f"w{i}" in joined for i in range(200))


def test_bad_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", size=100, overlap=80)


def test_build_chunks_keeps_source():
    chunks = build_chunks([("a.md", "alpha beta"), ("b.md", "gamma")], 100, 10)
    assert [c.source for c in chunks] == ["a.md", "b.md"]
