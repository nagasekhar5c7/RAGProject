"""
ingestion_pipeline.py — Orchestration

Responsibility:
    Wire all four pipeline components together and run them in sequence.
    This file contains zero business logic — it is a pure coordinator.

Pipeline steps (in order):
    1. Load     — DocumentLoaderFactory.load_all(data_dir)   → list[Document]
    2. Chunk    — RecursiveCharacterChunkStrategy.split()    → list[Document]
    3. Embed    — EmbeddingService.embed()                   → list[list[float]]
    4. Store    — FAISSVectorStore.store()                   → FAISS index on disk

Input  : data_dir (str)   — path to the folder containing raw documents
Output : FAISS index saved to vectorstore/faiss_index/ at the project root

All components are constructor-injected so the pipeline is fully testable —
each component can be swapped with a mock without touching pipeline code.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
from config.config import (
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    VECTOR_DB_TYPE,
    VECTOR_DB_PERSIST_PATH,
    VECTOR_DB_BATCH_SIZE,
)

from langchain_core.documents import Document

from base_loader import DocumentLoaderFactory
from chunk_strategies import RecursiveCharacterChunkStrategy
from embedding_service import EmbeddingService
from vectordb_factory import VectorDBFactory, VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates the full RAG ingestion flow in four sequential steps:
    Load → Chunk → Embed → Store.

    All four components are injected at construction time. The pipeline
    never instantiates components internally — it only calls them in order
    and passes results between steps.

    Attributes:
        loader       (DocumentLoaderFactory):            Step 1 — loads raw files.
        chunker      (RecursiveCharacterChunkStrategy):  Step 2 — splits into chunks.
        embedder     (EmbeddingService):                 Step 3 — converts to vectors.
        vector_store (VectorStore):                      Step 4 — persists to FAISS.
    """

    def __init__(
        self,
        loader: DocumentLoaderFactory,
        chunker: RecursiveCharacterChunkStrategy,
        embedder: EmbeddingService,
        vector_store: VectorStore,
    ):
        """
        Args:
            loader       (DocumentLoaderFactory):           Scans directory and loads files.
            chunker      (RecursiveCharacterChunkStrategy): Splits documents into chunks.
            embedder     (EmbeddingService):                Embeds chunks into float vectors.
            vector_store (VectorStore):                     Stores vectors in FAISS on disk.
        """
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def run(self, data_dir: str) -> None:
        """
        Execute the full ingestion pipeline end-to-end.

        Data flow:
            data_dir
              → documents  (list[Document])       via loader.load_all()
              → chunks     (list[Document])       via chunker.split()
              → embeddings (list[list[float]])    via embedder.embed()
              → FAISS index on disk               via vector_store.store()

        Args:
            data_dir (str): Path to the folder containing raw documents
                            (PDF, TXT, CSV). Scanned recursively.

        Raises:
            ValueError:    If data_dir is empty or no documents could be loaded.
            RuntimeError:  If any pipeline step fails, with a message indicating
                           which step failed and the underlying error.
        """
        logger.info("=" * 55)
        logger.info("RAG Ingestion Pipeline — START")
        logger.info(f"Data directory : {Path(data_dir).resolve()}")
        logger.info("=" * 55)

        # ── Step 1: Load ──────────────────────────────────────
        try:
            logger.info("[Step 1/4] Loading documents...")
            documents: list[Document] = self.loader.load_all(data_dir)
            logger.info(f"[Step 1/4] Done. {len(documents)} document(s) loaded.")
        except Exception as e:
            raise RuntimeError(f"Pipeline failed at Step 1 (Load): {e}") from e

        # ── Step 2: Chunk ─────────────────────────────────────
        try:
            logger.info("[Step 2/4] Chunking documents...")
            chunks: list[Document] = self.chunker.split(documents)
            logger.info(f"[Step 2/4] Done. {len(chunks)} chunk(s) produced.")
        except Exception as e:
            raise RuntimeError(f"Pipeline failed at Step 2 (Chunk): {e}") from e

        # ── Step 3: Embed ─────────────────────────────────────
        try:
            logger.info("[Step 3/4] Embedding chunks...")
            embeddings: list[list[float]] = self.embedder.embed(chunks)
            logger.info(f"[Step 3/4] Done. {len(embeddings)} vector(s) generated.")
        except Exception as e:
            raise RuntimeError(f"Pipeline failed at Step 3 (Embed): {e}") from e

        # ── Step 4: Store ─────────────────────────────────────
        try:
            logger.info("[Step 4/4] Storing vectors in FAISS...")
            self.vector_store.store(chunks, embeddings)
            logger.info("[Step 4/4] Done. FAISS index saved to disk.")
        except Exception as e:
            raise RuntimeError(f"Pipeline failed at Step 4 (Store): {e}") from e

        logger.info("=" * 55)
        logger.info("RAG Ingestion Pipeline — COMPLETE")
        logger.info(
            f"Summary: {len(documents)} docs → "
            f"{len(chunks)} chunks → "
            f"{len(embeddings)} vectors → FAISS"
        )
        logger.info("=" * 55)


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run the full ingestion pipeline from the command line.

    Usage (from project root):
        uv run python src/app/rag_ingestion/ingestion_pipeline.py
        uv run python src/app/rag_ingestion/ingestion_pipeline.py data/
    """
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR

    # ── Wire all components ───────────────────────────────────
    loader = DocumentLoaderFactory()

    chunker = RecursiveCharacterChunkStrategy(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    embedder = EmbeddingService(
        model_name=EMBEDDING_MODEL,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    db_factory = VectorDBFactory()
    vector_store = db_factory.create(
        VECTOR_DB_TYPE,
        persist_path=VECTOR_DB_PERSIST_PATH,
        embedding_model=embedder._model,
        batch_size=VECTOR_DB_BATCH_SIZE,
    )

    # ── Run ───────────────────────────────────────────────────
    pipeline = IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )
    pipeline.run(data_dir)

    # ── Verify output ─────────────────────────────────────────
    print(f"\nFAISS index files:")
    for f in sorted(Path(VECTOR_DB_PERSIST_PATH).glob("*")):
        print(f"  {f}  ({f.stat().st_size / 1024:.1f} KB)")
