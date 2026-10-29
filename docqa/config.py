"""Central configuration. Change values here, not in the code."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # runs locally, no API key
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 4             # chunks retrieved per question

LLM_PROVIDER = os.getenv("DOCQA_PROVIDER", "ollama").lower()
LLM_MODEL = os.getenv("DOCQA_MODEL", "llama3.2:3b" if LLM_PROVIDER == "ollama" else "claude-sonnet-4-5")
