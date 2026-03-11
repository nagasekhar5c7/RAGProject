"""
embedding_service.py — Step 3: Embedding

Responsibility:
    Convert a list of chunk Documents into a parallel list of float vectors
    using a HuggingFace Sentence Transformer model.

Model:
    sentence-transformers/all-MiniLM-L6-v2
        - Vector dimension : 384
        - Device           : CPU (no GPU dependency)
        - Fast inference, strong general-purpose semantic similarity

Input  : list[Document]       — chunks from chunk_strategies.py
Output : list[list[float]]    — one 384-dim vector per chunk

Output contract:
    len(output) == len(chunks)  — always guaranteed
    output[i]   corresponds to chunks[i]  — positional correspondence preserved
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
from config.config import EMBEDDING_MODEL, VECTOR_DIMENSION, EMBEDDING_BATCH_SIZE

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Embedding Service
# ──────────────────────────────────────────────

class EmbeddingService:
    """
    Converts chunk Documents into float vectors using a HuggingFace
    Sentence Transformer model.

    The model is loaded once at construction time — not per call — to avoid
    repeated cold-start latency. Embedding is processed in configurable
    batches to prevent memory exhaustion on large corpora.

    Attributes:
        model_name (str):  HuggingFace model identifier.
        batch_size (int):  Number of chunks embedded per batch.
        vector_dim (int):  Expected output vector dimension (384 for all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        """
        Load the HuggingFace embedding model once at init time.

        Args:
            model_name (str): HuggingFace model ID. Defaults to
                              'sentence-transformers/all-MiniLM-L6-v2'.
            batch_size (int): Chunks to embed per batch. Defaults to 32.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.vector_dim = VECTOR_DIMENSION

        logger.info(f"Loading embedding model: '{model_name}' (CPU) ...")
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(
            f"Embedding model loaded. "
            f"vector_dim={self.vector_dim}, batch_size={self.batch_size}"
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a single batch of raw text strings.

        Args:
            texts (list[str]): A batch of text strings to embed.

        Returns:
            list[list[float]]: One 384-dim float vector per input string.
        """
        return self._model.embed_documents(texts)

    def embed(self, chunks: list[Document]) -> list[list[float]]:
        """
        Embed all chunk Documents and return one vector per chunk.

        Processes chunks in batches of `batch_size` to keep memory usage
        bounded regardless of corpus size. The positional correspondence
        between input chunks and output vectors is always preserved:
        output[i] is the embedding for chunks[i].

        Args:
            chunks (list[Document]): Chunk Documents from the chunking step.

        Returns:
            list[list[float]]: Flat list of 384-dim float vectors,
                one per input chunk, in the same order.

        Raises:
            ValueError: If the chunks list is empty.
            AssertionError: If the number of embeddings does not match the
                number of input chunks (positional guarantee violated).
        """
        if not chunks:
            raise ValueError("No chunks provided to embed.")

        logger.info(f"Embedding {len(chunks)} chunk(s) in batches of {self.batch_size}...")

        texts = [chunk.page_content for chunk in chunks]
        embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            logger.info(f"  Batch {batch_num}/{total_batches} — {len(batch)} chunk(s)")
            batch_vectors = self._embed_batch(batch)
            embeddings.extend(batch_vectors)

        assert len(embeddings) == len(chunks), (
            f"Embedding count mismatch: got {len(embeddings)} vectors "
            f"for {len(chunks)} chunks."
        )

        logger.info(
            f"Embedding complete. "
            f"Total vectors: {len(embeddings)}, dimension: {self.vector_dim}"
        )
        return embeddings


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test. Loads → chunks → embeds documents from data/.

    Run from the project root:
        uv run python src/app/rag_ingestion/embedding_service.py
        uv run python src/app/rag_ingestion/embedding_service.py data/
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # project root

    from base_loader import DocumentLoaderFactory
    from chunk_strategies import RecursiveCharacterChunkStrategy

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    # Step 1: Load
    factory = DocumentLoaderFactory()
    documents = factory.load_all(data_dir)

    # Step 2: Chunk
    chunker = RecursiveCharacterChunkStrategy(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split(documents)

    # Step 3: Embed
    embedder = EmbeddingService()
    embeddings = embedder.embed(chunks)

    print(f"\n{'='*50}")
    print(f"Documents loaded  : {len(documents)}")
    print(f"Chunks produced   : {len(chunks)}")
    print(f"Vectors generated : {len(embeddings)}")
    print(f"Vector dimension  : {len(embeddings[0])}")
    print(f"{'='*50}")

    for i in range(min(3, len(embeddings))):
        print(f"\n[Vector {i+1}]")
        print(f"  Chunk source : {chunks[i].metadata.get('source', 'N/A')}")
        print(f"  Chunk index  : {chunks[i].metadata.get('chunk_index')} / {chunks[i].metadata.get('chunk_total')}")
        print(f"  Vector[:5]   : {embeddings[i][:5]}")
