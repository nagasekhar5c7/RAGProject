"""
reranker.py — Step 3: Cohere Reranking

Responsibility:
    Accept the ~10 candidate chunks returned by retriever.py, send them
    together with the original query to the Cohere Rerank API, and return
    the top-N chunks sorted by Cohere relevance score (highest first).

Why rerank?
    Vector similarity (cosine / L2) measures embedding proximity, not true
    semantic relevance.  A cross-encoder reranker (like Cohere Rerank) reads
    the full (query, chunk) pair and produces a calibrated relevance score —
    typically more accurate than embedding-only retrieval.

Input  : query  (str)           — the rewritten query from query_understanding.py
         chunks (list[Document])— candidate chunks from retriever.py (up to top_k)
Output : list[Document]         — top-N chunks sorted by relevance_score (desc)

Cohere relevance score:
    Injected into each returned Document's metadata as:
        relevance_score (float) — higher is more relevant (range varies by model)

Config:
    RERANKER_TOP_N  = 5                     (chunks returned after reranking)
    RERANKER_MODEL  = "rerank-english-v3.0" (Cohere reranker model)
    COHERE_API_KEY  = ""                    (set in config/config.py)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
from config.config import (
    COHERE_API_KEY,
    RERANKER_MODEL,
    RERANKER_TOP_N,
)

from langchain_cohere import CohereRerank
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────

class Reranker:
    """
    Reranks candidate chunks using the Cohere Rerank cross-encoder API.

    A cross-encoder evaluates each (query, chunk) pair jointly, producing a
    relevance score that is more accurate than the embedding-based similarity
    score used during retrieval.

    The top-N chunks by Cohere relevance score are returned, with the score
    injected into each chunk's metadata under the key ``relevance_score``.

    Attributes:
        top_n (int):   Number of top chunks to return after reranking.
        model (str):   Cohere reranker model name.
    """

    def __init__(
        self,
        top_n: int = RERANKER_TOP_N,
        model: str = RERANKER_MODEL,
        api_key: str = COHERE_API_KEY,
    ):
        """
        Initialise the Cohere reranker client.

        Args:
            top_n (int):    Number of chunks to return. Defaults to RERANKER_TOP_N.
            model (str):    Cohere reranker model. Defaults to RERANKER_MODEL.
            api_key (str):  Cohere API key. Defaults to COHERE_API_KEY from config.

        Raises:
            ValueError:   If api_key is empty.
            RuntimeError: If the Cohere client cannot be initialised.
        """
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY is not set. "
                "Add your key to config/config.py or set the COHERE_API_KEY variable."
            )

        self.top_n = top_n
        self.model = model

        logger.info(
            f"Initialising Cohere reranker — model='{model}', top_n={top_n}"
        )

        try:
            self._reranker = CohereRerank(
                cohere_api_key=api_key,
                model=model,
                top_n=top_n,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialise CohereRerank: {e}"
            ) from e

        logger.info("Cohere reranker initialised successfully.")

    def rerank(self, query: str, chunks: list[Document]) -> list[Document]:
        """
        Rerank candidate chunks against the query using Cohere Rerank.

        Steps:
            1. Validate the query and chunk list.
            2. Send all chunks to the Cohere Rerank API via compress_documents().
            3. Cohere selects the top-N most relevant chunks and injects
               ``relevance_score`` into each chunk's metadata.
            4. Return the top-N chunks sorted by relevance_score (highest first).

        Args:
            query  (str):           The rewritten query from query_understanding.py.
            chunks (list[Document]):Candidate chunks from retriever.py.

        Returns:
            list[Document]: Top-N reranked chunks with metadata:
                - relevance_score (float): Cohere cross-encoder relevance score.
                - All original metadata from retrieval is preserved.

        Raises:
            ValueError:   If the query is empty or no chunks are provided.
            RuntimeError: If the Cohere API call fails.
        """
        if not query.strip():
            raise ValueError("Query is empty — cannot rerank.")

        if not chunks:
            logger.warning("No chunks provided to reranker. Returning empty list.")
            return []

        logger.info(
            f"Reranking {len(chunks)} chunk(s) → keeping top {self.top_n}. "
            f"Query: '{query[:80]}{'...' if len(query) > 80 else ''}'"
        )

        try:
            reranked: list[Document] = list(
                self._reranker.compress_documents(chunks, query)
            )
        except Exception as e:
            raise RuntimeError(f"Cohere reranking failed: {e}") from e

        # Sort descending by relevance_score (Cohere already returns top_n in order,
        # but we sort explicitly to guarantee the contract regardless of API changes)
        reranked.sort(
            key=lambda d: d.metadata.get("relevance_score", 0.0),
            reverse=True,
        )

        logger.info(
            f"Reranking complete. Returning {len(reranked)} chunk(s) "
            f"(top relevance_score="
            f"{reranked[0].metadata.get('relevance_score', 'N/A') if reranked else 'N/A'})."
        )

        return reranked


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test.  Requires:
        1. A valid COHERE_API_KEY in config/config.py.
        2. A built FAISS index (run ingestion pipeline first).

    Run from the project root:
        uv run python src/app/rag_pipeline/reranker.py
    """
    from llm_service import LLMService
    from query_understanding import QueryUnderstanding
    from retriever import Retriever

    raw_query = "what is the attention mechanism in transformers?"

    # Step 1 — clean & rewrite
    qu = QueryUnderstanding(llm=LLMService())
    rewritten_query = qu.rewrite(raw_query)

    # Step 2 — retrieve
    retriever = Retriever()
    chunks = retriever.retrieve(rewritten_query)

    print(f"\n{'='*60}")
    print(f"Raw query      : {raw_query}")
    print(f"Rewritten query: {rewritten_query}")
    print(f"Chunks from retriever : {len(chunks)}")
    print(f"{'='*60}")

    # Step 3 — rerank
    reranker = Reranker()
    top_chunks = reranker.rerank(rewritten_query, chunks)

    print(f"\nTop {len(top_chunks)} chunk(s) after reranking:")
    print(f"{'='*60}")

    for i, doc in enumerate(top_chunks):
        print(f"\n[Chunk {i + 1}]")
        print(f"  Relevance score : {doc.metadata.get('relevance_score', 'N/A')}")
        print(f"  Similarity score: {doc.metadata.get('similarity_score', 'N/A')}")
        print(f"  Source          : {doc.metadata.get('source', 'N/A')}")
        print(f"  Page/Row        : {doc.metadata.get('page', doc.metadata.get('row', 'N/A'))}")
        print(f"  Content         : {doc.page_content[:120].strip()}...")
