"""
conftest.py — Shared pytest fixtures for BookRAG tests.

All tests use mocks/fakes — no API keys, no FAISS index, no network needed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Path setup so tests can import project modules ─────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RAG_PIPELINE_DIR = _PROJECT_ROOT / "src" / "app" / "rag_pipeline"
_RAG_INGESTION_DIR = _PROJECT_ROOT / "src" / "app" / "rag_ingestion"
_API_DIR = _PROJECT_ROOT / "src" / "api"

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_RAG_PIPELINE_DIR))
sys.path.insert(0, str(_RAG_INGESTION_DIR))
sys.path.insert(0, str(_API_DIR))

from langchain_core.documents import Document


# ── Fake LLMService ────────────────────────────────────────────

class FakeLLMService:
    """
    A fake LLMService that returns a canned response.
    No API key, no network call.
    """

    def __init__(self, response: str = "fake llm response"):
        self.response = response
        self.last_messages = None

    def generate(self, messages: list[dict]) -> str:
        self.last_messages = messages
        return self.response


@pytest.fixture
def fake_llm():
    """Returns a FakeLLMService that echoes a canned response."""
    return FakeLLMService(response="what is attention mechanism in transformers?")


@pytest.fixture
def sample_chunks():
    """Returns a list of 5 sample Document chunks with realistic metadata."""
    return [
        Document(
            page_content="The attention mechanism allows the model to focus on relevant parts of the input sequence.",
            metadata={"source": "transformers.pdf", "page": 12, "similarity_score": 0.85, "relevance_score": 0.92},
        ),
        Document(
            page_content="Self-attention computes a weighted sum of all positions in a sequence.",
            metadata={"source": "transformers.pdf", "page": 15, "similarity_score": 0.78, "relevance_score": 0.88},
        ),
        Document(
            page_content="The transformer architecture was introduced in the paper Attention Is All You Need.",
            metadata={"source": "deep_learning.pdf", "page": 42, "similarity_score": 0.72, "relevance_score": 0.80},
        ),
        Document(
            page_content="Multi-head attention runs several attention functions in parallel.",
            metadata={"source": "transformers.pdf", "page": 18, "similarity_score": 0.68, "relevance_score": 0.75},
        ),
        Document(
            page_content="Positional encoding adds information about the position of tokens in the sequence.",
            metadata={"source": "deep_learning.pdf", "page": 44, "similarity_score": 0.60, "relevance_score": 0.65},
        ),
    ]


@pytest.fixture
def duplicate_chunks():
    """Returns chunks where two have identical content (for dedup testing)."""
    return [
        Document(
            page_content="The attention mechanism allows the model to focus on relevant parts.",
            metadata={"source": "book_a.pdf", "page": 1, "relevance_score": 0.90},
        ),
        Document(
            page_content="The attention mechanism allows the model to focus on relevant parts.",
            metadata={"source": "book_b.pdf", "page": 5, "relevance_score": 0.85},
        ),
        Document(
            page_content="Transformers use self-attention to process sequences.",
            metadata={"source": "book_c.pdf", "page": 10, "relevance_score": 0.70},
        ),
    ]
