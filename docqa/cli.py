import argparse

from .chunking import build_chunks, load_documents
from .config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, EMBED_MODEL, INDEX_DIR, TOP_K
from .store import VectorStore


def cmd_ingest(args):
    docs = load_documents(DOCS_DIR)
    if not docs:
        raise SystemExit(f"No .md/.txt files found in {DOCS_DIR}")
    chunks = build_chunks(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    VectorStore.build(chunks, EMBED_MODEL).save(INDEX_DIR)
    print(f"Indexed {len(docs)} documents -> {len(chunks)} chunks in {INDEX_DIR}")


def cmd_ask(args):
    from .qa import answer
    store = VectorStore.load(INDEX_DIR)
    result = answer(args.question, store, k=args.k)
    print("\nAnswer:\n" + result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        preview = s["text"][:90].replace("\n", " ")
        print(f"  [{s['n']}] {s['source']} (chunk {s['chunk_id']}, score {s['score']}): {preview}...")


def main():
    p = argparse.ArgumentParser(prog="docqa", description="Document Q&A with source retrieval")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="chunk, embed and index data/docs").set_defaults(fn=cmd_ingest)
    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=TOP_K, help="chunks to retrieve")
    ask.set_defaults(fn=cmd_ask)
    args = p.parse_args()
    args.fn(args)
