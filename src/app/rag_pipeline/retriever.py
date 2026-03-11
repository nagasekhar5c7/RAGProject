"""
retriever.py — Step 2: FAISS Retrieval

Responsibility:
    Load the persisted FAISS index, run a similarity search against the
    rewritten query, convert L2 distances to [0, 1] similarity scores,
    and return only high-confidence chunks above the score threshold.

Input  : query (str)          — rewritten query from query_understanding.py
Output : list[Document]       — top-k chunks with similarity_score >= threshold

L2 → Similarity conversion:
    FAISS stores and returns L2 (Euclidean) distances.
    Lower L2 = more similar. We convert using:

        similarity = 1 / (1 + l2_distance)

    This maps:
        l2 = 0.0  →  similarity = 1.0  (perfect match)
        l2 = 1.0  →  similarity = 0.5
        l2 = ∞    →  similarity → 0.0

    If conversion fails for any chunk, the raw L2 distance is kept as-is
    and logged — the chunk is still included so no results are silently lost.

Config:
    RETRIEVER_TOP_K          = 10    (max chunks fetched from FAISS)
    RETRIEVER_SCORE_THRESHOLD = 0.5  (min similarity to keep a chunk)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
from config.config import (
    VECTOR_DB_PERSIST_PATH,
    EMBEDDING_MODEL,
    RETRIEVER_TOP_K,
    RETRIEVER_SCORE_THRESHOLD,
)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────

class Retriever:
    """
    Loads a persisted FAISS index and retrieves the most relevant chunks
    for a given query.

    FAISS returns L2 distances. These are converted to [0, 1] similarity
    scores using: similarity = 1 / (1 + l2_distance).

    Only chunks whose similarity score meets RETRIEVER_SCORE_THRESHOLD
    are returned. At most RETRIEVER_TOP_K chunks are returned.

    Attributes:
        top_k (int):            Maximum number of chunks to retrieve.
        score_threshold (float): Minimum similarity score (0-1) to keep a chunk.
        index_path (str):       Path to the persisted FAISS index directory.
    """

    def __init__(
        self,
        index_path: str = VECTOR_DB_PERSIST_PATH,
        top_k: int = RETRIEVER_TOP_K,
        score_threshold: float = RETRIEVER_SCORE_THRESHOLD,
        embedding_model_name: str = EMBEDDING_MODEL,
    ):
        """
        Load the FAISS index from disk at construction time.

        Args:
            index_path (str):           Path to the saved FAISS index directory.
            top_k (int):                Max chunks to fetch. Defaults to 10.
            score_threshold (float):    Min similarity score (0-1). Defaults to 0.5.
            embedding_model_name (str): HuggingFace model used during ingestion.
                                        Must match what was used to build the index.

        Raises:
            RuntimeError: If the FAISS index cannot be loaded from disk.
        """
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.index_path = index_path

        logger.info(f"Loading embedding model '{embedding_model_name}' for retrieval...")
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"Loading FAISS index from '{index_path}'...")
        try:
            self._index = FAISS.load_local(
                index_path,
                self._embedding_model,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FAISS index from '{index_path}': {e}\n"
                "Run the ingestion pipeline first to build the index."
            ) from e

    def _l2_to_similarity(self, l2_distance: float) -> float:
        """
        Convert an L2 distance to a [0, 1] similarity score.

        Formula: similarity = 1 / (1 + l2_distance)

        Args:
            l2_distance (float): L2 distance returned by FAISS (>= 0).

        Returns:
            float: Similarity score in [0, 1]. Higher = more similar.
        """
        return 1.0 / (1.0 + l2_distance)

    def retrieve(self, query: str) -> list[Document]:
        """
        Search the FAISS index for chunks most similar to the query.

        Steps:
            1. Fetch top_k candidates from FAISS with L2 distances.
            2. Convert each L2 distance to a [0, 1] similarity score.
            3. Inject similarity_score into each chunk's metadata.
            4. Filter out chunks below score_threshold.
            5. Return remaining chunks ordered by score (highest first).

        Args:
            query (str): The rewritten query from query_understanding.py.

        Returns:
            list[Document]: High-confidence chunks with metadata:
                - similarity_score (float): converted score in [0, 1]
                - l2_distance (float):      raw FAISS L2 distance
                - All original metadata from ingestion is preserved.

        Raises:
            ValueError: If the query is empty.
            RuntimeError: If the FAISS search fails.
        """
        if not query.strip():
            raise ValueError("Query is empty — cannot retrieve.")

        logger.info(
            f"Retrieving top {self.top_k} chunks for query: "
            f"'{query[:80]}{'...' if len(query) > 80 else ''}'"
        )

        try:
            results: list[tuple[Document, float]] = (
                self._index.similarity_search_with_score(query, k=self.top_k)
            )
        except Exception as e:
            raise RuntimeError(f"FAISS search failed: {e}") from e

        logger.info(f"FAISS returned {len(results)} candidate(s). Applying score filter...")

        scored_chunks: list[Document] = []

        for doc, l2_distance in results:
            try:
                similarity = self._l2_to_similarity(l2_distance)
            except Exception as e:
                # Fallback: keep raw L2, do not drop the chunk silently
                logger.warning(
                    f"Score conversion failed for chunk "
                    f"(source={doc.metadata.get('source', '?')}): {e}. "
                    f"Using raw L2={l2_distance:.4f} as score."
                )
                similarity = l2_distance

            doc.metadata["similarity_score"] = round(similarity, 4)
            doc.metadata["l2_distance"] = round(l2_distance, 4)

            if similarity >= self.score_threshold:
                scored_chunks.append(doc)
            else:
                logger.debug(
                    f"Chunk filtered out — score={similarity:.4f} < "
                    f"threshold={self.score_threshold} "
                    f"(source={doc.metadata.get('source', '?')})"
                )

        # Sort by similarity descending
        scored_chunks.sort(key=lambda d: d.metadata["similarity_score"], reverse=True)

        logger.info(
            f"Retrieval complete. "
            f"{len(scored_chunks)}/{len(results)} chunk(s) passed "
            f"score threshold ({self.score_threshold})."
        )
        return scored_chunks


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test. Requires a built FAISS index at vectorstore/faiss_index/.
    Run the ingestion pipeline first if the index doesn't exist.

    Run from the project root:
        uv run python src/app/rag_pipeline/retriever.py
    """
    query = "what is the attention mechanism in transformers?"

    retriever = Retriever()
    chunks = retriever.retrieve(query)

    print(f"\n{'='*55}")
    print(f"Query   : {query}")
    print(f"Chunks returned : {len(chunks)}")
    print(f"{'='*55}")

    for i, doc in enumerate(chunks):
        print(f"\n[Chunk {i+1}]")
        print(f"  Score     : {doc.metadata.get('similarity_score')}")
        print(f"  L2 dist   : {doc.metadata.get('l2_distance')}")
        print(f"  Source    : {doc.metadata.get('source', 'N/A')}")
        print(f"  Page/Row  : {doc.metadata.get('page', doc.metadata.get('row', 'N/A'))}")
        print(f"  Content   : {doc.page_content[:120].strip()}...")
