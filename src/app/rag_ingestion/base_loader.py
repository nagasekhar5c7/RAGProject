"""
base_loader.py — Step 1: Document Loading

Responsibility:
    Scan a directory, detect file types, dispatch to the correct LangChain
    loader, and return a unified List[Document] with standardized metadata.

Supported file types:
    - PDF  (.pdf)  → loaded per page via PyMuPDFLoader
    - TXT  (.txt)  → loaded per file via LangChain TextLoader
    - CSV  (.csv)  → loaded per row  via LangChain CSVLoader

Output contract:
    Every Document returned carries:
        metadata["source"]    — absolute path to the source file
        metadata["file_type"] — "pdf" | "txt" | "csv"
        metadata["page"]      — page number (PDF only)
        metadata["row"]       — row index  (CSV only)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────

class BaseDocumentLoader(ABC):
    """
    Abstract base class for all document loaders.

    Enforces a common interface: every concrete loader must accept a file path
    at construction time and expose a single `load()` method that returns a
    list of LangChain Document objects.

    Attributes:
        file_path (str): Absolute or relative path to the file to be loaded.
    """

    def __init__(self, file_path: str):
        """
        Args:
            file_path (str): Path to the file this loader will process.
        """
        self.file_path = file_path

    @abstractmethod
    def load(self) -> list[Document]:
        """
        Load the file and return its contents as a list of Documents.

        Returns:
            list[Document]: One or more Documents with page_content and metadata.

        Raises:
            Exception: If the file cannot be read or parsed.
        """
        pass


# ──────────────────────────────────────────────
# Concrete Loaders
# ──────────────────────────────────────────────

class PDFDocumentLoader(BaseDocumentLoader):
    """
    Loads PDF files using PyMuPDFLoader (faster and more robust than PyPDFLoader).

    Each page of the PDF becomes a separate Document. Page numbers are preserved
    in metadata via the 'page' key (set automatically by PyMuPDFLoader).
    """

    def load(self) -> list[Document]:
        """
        Load a PDF file page-by-page.

        Returns:
            list[Document]: One Document per page, each with metadata:
                - source    : file path
                - file_type : "pdf"
                - page      : 0-based page index (set by PyMuPDFLoader)
        """
        loader = PyMuPDFLoader(self.file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = self.file_path
            doc.metadata["file_type"] = "pdf"
            # PyMuPDFLoader already sets 'page' in metadata
        logger.info(f"PDF: loaded {len(docs)} page(s) from {self.file_path}")
        return docs


class TextDocumentLoader(BaseDocumentLoader):
    """
    Loads plain text files (.txt) as a single Document per file.

    Uses UTF-8 encoding with autodetect fallback to handle files with
    non-standard encodings gracefully (avoids UnicodeDecodeError crashes).
    """

    def load(self) -> list[Document]:
        """
        Load a plain text file as a single Document.

        Returns:
            list[Document]: One Document containing the full file content, with metadata:
                - source    : file path
                - file_type : "txt"
        """
        loader = TextLoader(self.file_path, encoding="utf-8", autodetect_encoding=True)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = self.file_path
            doc.metadata["file_type"] = "txt"
        logger.info(f"TXT: loaded {len(docs)} document(s) from {self.file_path}")
        return docs


class CSVDocumentLoader(BaseDocumentLoader):
    """
    Loads CSV files where each row becomes a separate Document.

    Row index is injected into metadata so downstream components can trace
    exactly which row a chunk originated from.
    """

    def load(self) -> list[Document]:
        """
        Load a CSV file row-by-row.

        Returns:
            list[Document]: One Document per data row, each with metadata:
                - source    : file path
                - file_type : "csv"
                - row       : 0-based row index
        """
        loader = CSVLoader(self.file_path)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["source"] = self.file_path
            doc.metadata["file_type"] = "csv"
            doc.metadata["row"] = i
        logger.info(f"CSV: loaded {len(docs)} row(s) from {self.file_path}")
        return docs


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

class DocumentLoaderFactory:
    """
    Factory that maps file extensions to loader classes and orchestrates
    loading across an entire directory.

    Uses a registry pattern (dict of extension → loader class) instead of
    if/elif chains. Adding support for a new file type is a single
    `register()` call — no modifications to existing code required.

    Registry (default):
        ".pdf" → PDFDocumentLoader
        ".txt" → TextDocumentLoader
        ".csv" → CSVDocumentLoader
    """

    def __init__(self):
        """
        Initialize the factory and register all default loaders.
        """
        self._registry: dict[str, type[BaseDocumentLoader]] = {}
        # Register supported loaders
        self.register(".pdf", PDFDocumentLoader)
        self.register(".txt", TextDocumentLoader)
        self.register(".csv", CSVDocumentLoader)

    def register(self, extension: str, loader_class: type[BaseDocumentLoader]) -> None:
        """
        Register a loader class for a given file extension.

        Args:
            extension (str):              File extension including the dot, e.g. ".pdf".
            loader_class (type):          A subclass of BaseDocumentLoader to handle this type.
        """
        self._registry[extension.lower()] = loader_class
        logger.debug(f"Registered loader for '{extension}': {loader_class.__name__}")

    def get_loader(self, file_path: str) -> BaseDocumentLoader | None:
        """
        Look up and instantiate the correct loader for a given file.

        Args:
            file_path (str): Path to the file whose loader should be retrieved.

        Returns:
            BaseDocumentLoader | None: An instantiated loader, or None if the
                file extension is not in the registry (unsupported type).
        """
        ext = Path(file_path).suffix.lower()
        loader_class = self._registry.get(ext)
        if loader_class is None:
            logger.warning(f"Unsupported file type '{ext}' — skipping: {file_path}")
            return None
        return loader_class(file_path)

    def load_all(self, directory: str) -> list[Document]:
        """
        Recursively scan a directory and load all supported files.

        Walks the entire directory tree, dispatches each file to the appropriate
        loader, and returns a flat list of all Documents. Unsupported file types
        are skipped with a warning. Corrupt or unreadable files are logged and
        skipped without stopping the pipeline.

        Args:
            directory (str): Path to the folder containing raw documents.

        Returns:
            list[Document]: All loaded Documents from all supported files,
                each carrying standardized metadata (source, file_type, page/row).

        Raises:
            ValueError: If the directory does not exist.
            ValueError: If the directory contains no files at all.
            ValueError: If zero Documents were successfully loaded (all files
                were unsupported or failed to parse).
        """
        data_dir = Path(directory)

        if not data_dir.exists() or not data_dir.is_dir():
            raise ValueError(f"Directory not found: {directory}")

        # Recursively scan all files
        all_files = [f for f in data_dir.rglob("*") if f.is_file()]

        if not all_files:
            raise ValueError(f"No files found in directory: {directory}")

        logger.info(f"Found {len(all_files)} file(s) in '{directory}'")

        all_documents: list[Document] = []

        for file_path in all_files:
            loader = self.get_loader(str(file_path))
            if loader is None:
                continue  # unsupported type — already logged
            try:
                docs = loader.load()
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load '{file_path}': {e}")
                continue  # skip corrupt/unreadable files

        if not all_documents:
            raise ValueError(
                f"Zero documents loaded from '{directory}'. "
                "Check that the folder contains supported files (PDF, TXT, CSV)."
            )

        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test. Run from the project root:

        python -m src.app.rag_ingestion.base_loader data/
        # or pass a custom path:
        python -m src.app.rag_ingestion.base_loader /path/to/data

    Prints a summary of every loaded document: source, file type,
    page/row number, and the first 100 characters of content.
    """
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    factory = DocumentLoaderFactory()
    documents = factory.load_all(data_dir)

    print(f"\n{'='*50}")
    print(f"Total documents loaded: {len(documents)}")
    print(f"{'='*50}")

    for i, doc in enumerate(documents):
        print(f"\n[Doc {i+1}]")
        print(f"  Source    : {doc.metadata.get('source', 'N/A')}")
        print(f"  File Type : {doc.metadata.get('file_type', 'N/A')}")
        print(f"  Page/Row  : {doc.metadata.get('page', doc.metadata.get('row', 'N/A'))}")
        print(f"  Content   : {doc.page_content[:100].strip()}...")
