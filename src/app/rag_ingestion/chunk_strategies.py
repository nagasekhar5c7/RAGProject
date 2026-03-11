"""
chunk_strategies.py — Step 2: Document Chunking

Responsibility:
    Accept a list of loaded Documents and return a flat list of overlapping
    text chunks, with all source metadata preserved and chunk position
    metadata injected.

Strategy:
    RecursiveCharacterChunkStrategy — splits on natural text boundaries
    (paragraphs → sentences → words → characters) using LangChain's
    RecursiveCharacterTextSplitter.

    chunk_size:    1000 characters (max characters per chunk)
    chunk_overlap: 200  characters (overlap between consecutive chunks)

Output contract:
    Every chunk Document carries:
        metadata["source"]      — inherited from the original document
        metadata["file_type"]   — inherited from the original document
        metadata["page"]        — inherited (PDF only)
        metadata["row"]         — inherited (CSV only)
        metadata["chunk_index"] — 0-based position of this chunk within its source document
        metadata["chunk_total"] — total number of chunks produced from that source document
"""

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────

class ChunkStrategy(ABC):
    """
    Abstract base class for all chunking strategies.

    Enforces a common interface: every concrete strategy must accept
    chunk_size and chunk_overlap at construction time and expose a single
    `split()` method that takes a list of Documents and returns a list of
    chunk Documents.

    Attributes:
        chunk_size (int):    Maximum number of characters per chunk.
        chunk_overlap (int): Number of characters shared between consecutive chunks.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Args:
            chunk_size (int):    Maximum characters per chunk.
            chunk_overlap (int): Characters of overlap between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of Documents into smaller chunk Documents.

        Args:
            documents (list[Document]): Full documents returned by the loader step.

        Returns:
            list[Document]: Flat list of chunk Documents, each with source
                metadata preserved and chunk_index / chunk_total injected.
        """
        pass


# ──────────────────────────────────────────────
# Concrete Strategy
# ──────────────────────────────────────────────

class RecursiveCharacterChunkStrategy(ChunkStrategy):
    """
    Fixed-size chunking strategy using LangChain's RecursiveCharacterTextSplitter.

    Splits text by attempting progressively smaller boundaries in this order:
        paragraph breaks (\\n\\n) → line breaks (\\n) → spaces → individual characters

    This ensures chunks respect natural text structure wherever possible
    before falling back to hard character splits.

    Each source Document is chunked independently, so chunk_index and
    chunk_total always refer to position within the originating document —
    not across the entire corpus.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Args:
            chunk_size (int):    Max characters per chunk. Defaults to 1000.
            chunk_overlap (int): Overlap between adjacent chunks. Defaults to 200.
        """
        super().__init__(chunk_size, chunk_overlap)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        logger.info(
            f"RecursiveCharacterChunkStrategy initialized "
            f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split each document independently and inject chunk position metadata.

        Processing steps per document:
            1. Run RecursiveCharacterTextSplitter on the document.
            2. Inject chunk_index (0-based) and chunk_total into each chunk's metadata.
            3. All existing metadata (source, file_type, page, row) is preserved
               automatically by LangChain's splitter.

        Args:
            documents (list[Document]): Loaded documents from the base_loader step.

        Returns:
            list[Document]: Flat list of all chunks across all documents,
                ordered by document then chunk position.

        Raises:
            ValueError: If the input document list is empty.
        """
        if not documents:
            raise ValueError("No documents provided to split.")

        logger.info(f"Splitting {len(documents)} document(s) into chunks...")

        all_chunks: list[Document] = []

        for doc in documents:
            # Split this single document into chunks
            chunks = self._splitter.split_documents([doc])

            total = len(chunks)

            # Inject chunk position metadata
            for index, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = index
                chunk.metadata["chunk_total"] = total

            all_chunks.extend(chunks)

            logger.debug(
                f"  {doc.metadata.get('source', 'unknown')} "
                f"[page/row {doc.metadata.get('page', doc.metadata.get('row', '-'))}] "
                f"→ {total} chunk(s)"
            )

        logger.info(f"Chunking complete. Total chunks produced: {len(all_chunks)}")
        return all_chunks


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test. Loads documents from data/ using DocumentLoaderFactory,
    then splits them using RecursiveCharacterChunkStrategy.

    Run from the project root:
        uv run python src/app/rag_ingestion/chunk_strategies.py
    """
    import sys
    from base_loader import DocumentLoaderFactory

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    # Step 1: Load
    factory = DocumentLoaderFactory()
    documents = factory.load_all(data_dir)

    # Step 2: Chunk
    chunker = RecursiveCharacterChunkStrategy(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split(documents)

    print(f"\n{'='*50}")
    print(f"Documents loaded : {len(documents)}")
    print(f"Chunks produced  : {len(chunks)}")
    print(f"{'='*50}")

    for i, chunk in enumerate(chunks[:5]):  # preview first 5 chunks
        print(f"\n[Chunk {i+1}]")
        print(f"  Source      : {chunk.metadata.get('source', 'N/A')}")
        print(f"  File Type   : {chunk.metadata.get('file_type', 'N/A')}")
        print(f"  Page/Row    : {chunk.metadata.get('page', chunk.metadata.get('row', 'N/A'))}")
        print(f"  Chunk Index : {chunk.metadata.get('chunk_index')} / {chunk.metadata.get('chunk_total')}")
        print(f"  Length      : {len(chunk.page_content)} chars")
        print(f"  Content     : {chunk.page_content[:120].strip()}...")
