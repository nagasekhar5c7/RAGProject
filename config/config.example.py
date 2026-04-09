"""
config.example.py — Configuration Template

Copy this file to config.py and fill in your API keys:
    cp config/config.example.py config/config.py

config.py is excluded from git (.gitignore) to prevent secrets from
being committed. This example file is committed as a reference.

API keys use os.environ.get() so they can be injected via environment
variables in Kubernetes without changing this file.
"""

import os

# ── Data ──────────────────────────────────────────────────────
DATA_DIR = "data"

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Embedding ─────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32

# ── Vector Store ──────────────────────────────────────────────
VECTOR_DB_TYPE = "faiss"
VECTOR_DB_PERSIST_PATH = "vectorstore/faiss_index"
VECTOR_DB_BATCH_SIZE = 500

# ── Retriever ─────────────────────────────────────────────────
RETRIEVER_TOP_K = 10              # Max chunks to return from FAISS
RETRIEVER_SCORE_THRESHOLD = 0.5  # Min similarity score (0-1) to keep a chunk

# ── Reranker (Cohere) ─────────────────────────────────────────
RERANKER_TOP_N = 5                      # Top chunks to return after reranking
RERANKER_MODEL = "rerank-english-v3.0"  # Cohere reranker model
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")   # Get your key at: https://dashboard.cohere.com

# ── Context Builder ───────────────────────────────────────────
CONTEXT_MAX_TOKENS = 2000           # Max token budget for the context block
CONTEXT_CHUNK_SEPARATOR = "\n\n---\n\n"  # Separator between formatted chunks

# ── API (FastAPI / Uvicorn) ───────────────────────────────────
API_HOST = "0.0.0.0"   # Bind address
API_PORT = 8000         # Uvicorn port

# ── Observability (LangSmith) ─────────────────────────────────
LANGSMITH_API_KEY        = os.environ.get("LANGSMITH_API_KEY", "")   # Get your key at: https://smith.langchain.com
LANGSMITH_PROJECT        = "BookRAG"   # Project name shown in LangSmith UI
LANGSMITH_TRACING_ENABLED = True       # Set False to disable tracing

# ── LLM (Groq) ────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"   # Groq-hosted open-source model
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1024
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")      # Get your key at: https://console.groq.com
