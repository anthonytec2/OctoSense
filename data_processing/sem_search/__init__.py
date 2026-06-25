"""
Semantic Video Search — sub-second semantic search over large RGB robotics logs.

Two halves (see main.py for the CLI):
  - processing: build the index — caption generation (Gemma-4 via vLLM) + text
    embeddings (Qwen3-Embedding) -> FAISS index.
  - serving: FastAPI web UI + /search, and a CLI/Python query path.

Entry point:
    python -m sem_search.main {ingest-all|serve|query}
"""

__version__ = "1.0.0"
