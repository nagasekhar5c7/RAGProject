"""
vectordb_factory.py — Step 4: Vector Storage

Responsibility:
    Accept chunk Documents and their corresponding embeddings, write them
    into a FAISS index in batches, and persist the index to local disk.

Input:
    chunks     : list[Document]       — chunks from chunk_strategies.py
    embeddings : list[list[float]]    — vectors from embedding_service.py

Output:
    FAISS index saved to disk under:
        <project_root>/vectorstore/faiss_index/

    Two files are written by FAISS:
        faiss_index/index.faiss   — the binary vector index
        faiss_index/index.pkl     — the docstore (metadata + page_content)

Design:
    VectorStore         — abstract base class enforcing store() + persist()
    FAISSVectorStore    — concrete FAISS implementation with batch writes
    VectorDBFactory     — registry-based factory; swap backends via config
"""

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
from config.config import VECTOR_DB_PERSIST_PATH, VECTOR_DB_BATCH_SIZE

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────

class VectorStore(ABC):
    """
    Abstract base class for all vector store backends.

    Enforces a two-method contract on every concrete backend:
        store()   — write chunks + embeddings into the index
        persist() — flush the index to durable storage
    """

    @abstractmethod
    def store(self, chunks: list[Document], embeddings: list[list[float]]) -> None:
        """
        Write chunk Documents and their embeddings into the vector index.

        Args:
            chunks     (list[Document]):      Chunk Documents with metadata.
            embeddings (list[list[float]]):   One float vector per chunk,
                                              positionally aligned with chunks.
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """
        Persist the vector index to durable storage (disk, cloud, etc.).
        """
        pass


# ──────────────────────────────────────────────
# Concrete FAISS Implementation
# ──────────────────────────────────────────────

class FAISSVectorStore(VectorStore):
    """
    FAISS-backed vector store that writes embeddings in batches and saves
    the index to local disk via LangChain's FAISS.save_local().

    Batch writes keep memory usage bounded for large corpora. The first
    batch bootstraps the index; subsequent batches merge into it.

    persist() is called once after all batches complete — not after each
    batch — to avoid redundant disk I/O.

    Saved files (under persist_path/):
        index.faiss   — binary FAISS vector index
        index.pkl     — docstore mapping (metadata + page_content)

    Attributes:
        persist_path (str):  Directory where the FAISS index is saved.
        batch_size (int):    Number of chunk/vector pairs written per batch.
    """

    def __init__(
        self,
        persist_path: str = VECTOR_DB_PERSIST_PATH,
        embedding_model: HuggingFaceEmbeddings = None,
        batch_size: int = VECTOR_DB_BATCH_SIZE,
    ):
        """
        Args:
            persist_path (str):                  Directory path for saving the index.
            embedding_model (HuggingFaceEmbeddings): The same embedding model used
                                                 to produce the vectors — required
                                                 by LangChain's FAISS wrapper for
                                                 future similarity searches.
            batch_size (int):                    Chunk/vector pairs per write batch.
                                                 Defaults to 500.
        """
        self.persist_path = persist_path
        self.batch_size = batch_size
        self._embedding_model = embedding_model
        self._index: FAISS | None = None

        Path(persist_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"FAISSVectorStore initialized. persist_path='{persist_path}'")

    def store(self, chunks: list[Document], embeddings: list[list[float]]) -> None:
        """
        Write all chunks and embeddings into the FAISS index in batches.

        The first batch creates the index from scratch using
        FAISS.from_embeddings(). Every subsequent batch is merged in via
        add_embeddings(). persist() is called once after the loop.

        Args:
            chunks     (list[Document]):     Chunk Documents (metadata preserved).
            embeddings (list[list[float]]): One float vector per chunk.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )

        logger.info(
            f"Storing {len(chunks)} chunk(s) into FAISS "
            f"in batches of {self.batch_size}..."
        )

        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_embeddings = embeddings[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            # Pair each embedding with its text for FAISS
            text_embedding_pairs = [
                (chunk.page_content, embedding)
                for chunk, embedding in zip(batch_chunks, batch_embeddings)
            ]
            metadatas = [chunk.metadata for chunk in batch_chunks]

            if self._index is None:
                # First batch — bootstrap the index
                self._index = FAISS.from_embeddings(
                    text_embeddings=text_embedding_pairs,
                    embedding=self._embedding_model,
                    metadatas=metadatas,
                )
                logger.info(f"  Batch {batch_num}/{total_batches} — index created.")
            else:
                # Subsequent batches — merge into existing index
                self._index.add_embeddings(
                    text_embeddings=text_embedding_pairs,
                    metadatas=metadatas,
                )
                logger.info(f"  Batch {batch_num}/{total_batches} — merged into index.")

        self.persist()

    def persist(self) -> None:
        """
        Save the FAISS index to disk at persist_path.

        Writes two files:
            index.faiss  — binary vector index
            index.pkl    — docstore with metadata + page content

        Raises:
            RuntimeError: If persist() is called before any data has been stored.
        """
        if self._index is None:
            raise RuntimeError("Cannot persist: no data has been stored yet.")

        self._index.save_local(self.persist_path)
        logger.info(
            f"FAISS index persisted to '{self.persist_path}' "
            f"(index.faiss + index.pkl)"
        )


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

class VectorDBFactory:
    """
    Registry-based factory for creating vector store backends.

    Maps backend type strings (e.g. "faiss") to concrete VectorStore classes.
    Adding a new backend (Chroma, Pinecone, Weaviate) is a single register()
    call — no changes to the pipeline code required.

    Registry (default):
        "faiss" → FAISSVectorStore
    """

    def __init__(self):
        """
        Initialize the factory and register the default FAISS backend.
        """
        self._registry: dict[str, type[VectorStore]] = {}
        self.register("faiss", FAISSVectorStore)

    def register(self, db_type: str, store_class: type[VectorStore]) -> None:
        """
        Register a vector store class for a given backend type string.

        Args:
            db_type (str):             Identifier string, e.g. "faiss", "chroma".
            store_class (type):        A subclass of VectorStore.
        """
        self._registry[db_type.lower()] = store_class
        logger.debug(f"Registered vector store: '{db_type}' → {store_class.__name__}")

    def create(self, db_type: str, **kwargs) -> VectorStore:
        """
        Instantiate and return a vector store for the requested backend type.

        Args:
            db_type (str):   Backend identifier, e.g. "faiss".
            **kwargs:        Constructor arguments forwarded to the store class.

        Returns:
            VectorStore: A ready-to-use vector store instance.

        Raises:
            ValueError: If db_type is not registered.
        """
        store_class = self._registry.get(db_type.lower())
        if store_class is None:
            raise ValueError(
                f"Unknown vector DB type: '{db_type}'. "
                f"Available: {list(self._registry.keys())}"
            )
        logger.info(f"Creating vector store: '{db_type}'")
        return store_class(**kwargs)


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test. Runs the full pipeline up to vector storage:
        Load → Chunk → Embed → Store to FAISS

    Run from the project root:
        uv run python src/app/rag_ingestion/vectordb_factory.py
        uv run python src/app/rag_ingestion/vectordb_factory.py data/
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # project root

    from base_loader import DocumentLoaderFactory
    from chunk_strategies import RecursiveCharacterChunkStrategy
    from embedding_service import EmbeddingService

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    persist_path = "vectorstore/faiss_index"

    # Step 1: Load
    factory = DocumentLoaderFactory()
    documents = factory.load_all(data_dir)

    # Step 2: Chunk
    chunker = RecursiveCharacterChunkStrategy(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split(documents)

    # Step 3: Embed
    embedder = EmbeddingService()
    embeddings = embedder.embed(chunks)

    # Step 4: Store
    db_factory = VectorDBFactory()
    vector_store = db_factory.create(
        "faiss",
        persist_path=persist_path,
        embedding_model=embedder._model,
        batch_size=500,
    )
    vector_store.store(chunks, embeddings)

    print(f"\n{'='*50}")
    print(f"Documents loaded  : {len(documents)}")
    print(f"Chunks stored     : {len(chunks)}")
    print(f"Vectors stored    : {len(embeddings)}")
    print(f"FAISS index saved : {persist_path}/")
    print(f"{'='*50}")

    # Verify files exist on disk
    index_files = list(Path(persist_path).glob("*"))
    print(f"\nFiles written to disk:")
    for f in index_files:
        print(f"  {f}  ({f.stat().st_size / 1024:.1f} KB)")
