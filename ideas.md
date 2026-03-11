# RAG Ingestion Pipeline — Architecture & Design Analysis

## Project Context

**Project:** BookRAGProject
**Python Version:** 3.12
**Orchestration Framework:** LangChain
**Current State:** Bare scaffold — no dependencies, no src/, no data/ directory yet.

This document captures the full architectural analysis of the end-to-end RAG ingestion pipeline
before any implementation begins. It serves as the single source of truth for design decisions,
component contracts, patterns, risks, and implementation guidance.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Pipeline Execution Model](#2-pipeline-execution-model)
3. [Component Analysis](#3-component-analysis)
  - [base_loader.py](#31-base_loaderpy--document-loading)
  - [chunk_strategies.py](#32-chunk_strategiespy--chunking)
  - [embedding_service.py](#33-embedding_servicepy--embedding)
  - [vectordb_factory.py](#34-vectordb_factorypy--vector-storage)
  - [ingestion_pipeline.py](#35-ingestion_pipelinepy--orchestration)
4. [Design Patterns Used](#4-design-patterns-used)
5. [Cross-Cutting Concerns](#5-cross-cutting-concerns)
6. [Dependency Map](#6-dependency-map)
7. [Risk Register](#7-risk-register)
8. [Open Questions](#8-open-questions)

---

## 1. Pipeline Overview

The ingestion pipeline is a **linear DAG** with four transformation stages and one
orchestration layer. Data flows in one direction — from raw files on disk to persisted
vectors in FAISS — with no cycles or branches under the happy path.

```
data/
  └── (PDF, TXT, CSV, DOCX, XLSX, JSON)
         │
         ▼
  ┌─────────────┐
  │  base_loader │   Step 1 — Load & standardize documents
  └──────┬──────┘
         │  List[Document]
         ▼
  ┌──────────────────┐
  │ chunk_strategies  │   Step 2 — Split into overlapping chunks
  └────────┬─────────┘
           │  List[Document]  (chunks)
           ▼
  ┌──────────────────┐
  │ embedding_service │   Step 3 — Convert chunks → float vectors
  └────────┬─────────┘
           │  List[List[float]]
           ▼
  ┌─────────────────┐
  │ vectordb_factory │   Step 4 — Persist vectors to FAISS
  └─────────────────┘
           │
           ▼
      FAISS index (saved to disk)

  ──────────────────────────────────────────────
  ingestion_pipeline.py  →  plain LangChain
  orchestrator that calls each component in
  sequence and manages results between steps
```

Each component owns exactly one responsibility. The pipeline file owns zero business logic —
it only wires components together and passes results between steps.

---

## 2. Pipeline Execution Model

The pipeline uses **plain LangChain** with a straightforward sequential method-call pattern.
There are no graphs, no nodes, no state machines, and no conditional edge routing.
`IngestionPipeline.run()` calls each component in order, passing the output of one step
directly as the input to the next.

### Execution Flow

```
run(data_dir: str)
  │
  ├── 1. documents = loader.load_all(data_dir)
  │
  ├── 2. chunks = chunker.split(documents)
  │
  ├── 3. embeddings = embedder.embed(chunks)
  │
  └── 4. vector_store.store(chunks, embeddings)
            └── vector_store.persist()
```

### Data Passed Between Steps

| Step | Input | Output |
| ---- | ----- | ------ |
| Load | `data_dir: str` | `documents: List[Document]` |
| Chunk | `documents: List[Document]` | `chunks: List[Document]` |
| Embed | `chunks: List[Document]` | `embeddings: List[List[float]]` |
| Store | `chunks + embeddings` | FAISS index saved to disk |

### Error Handling

Standard Python `try/except` at the `run()` level. Each step raises on failure;
the pipeline catches, logs, and re-raises with a descriptive message indicating
which step failed. No error routing logic is needed.

---

## 3. Component Analysis

---

### 3.1 `base_loader.py` — Document Loading

**Single Responsibility:** Scan a directory, detect file types, dispatch to the correct
LangChain loader, and return a unified `List[Document]`.

#### Class Design

```
BaseDocumentLoader  (abstract base class)
├── __init__(file_path: str)
└── load() → List[Document]   ← abstract method

PDFLoader(BaseDocumentLoader)
TextLoader(BaseDocumentLoader)
CSVLoader(BaseDocumentLoader)

DocumentLoaderFactory
├── _registry: dict[str, type[BaseDocumentLoader]]
├── register(extension, loader_class)
├── get_loader(file_path) → BaseDocumentLoader
└── load_all(directory: str) → List[Document]
```

#### Loader Registry

The factory maps file extensions to loader classes. This avoids `if/elif` chains and makes
adding support for new file types a one-line change.

```
Registry (in scope):
  ".pdf"  → PDFLoader   (wraps PyMuPDFLoader)
  ".txt"  → TextLoader  (wraps LangChain TextLoader)
  ".csv"  → CSVLoader   (wraps LangChain CSVLoader)
```

DOCX, XLSX, and JSON loaders are **deferred** — out of scope for the initial implementation.
Future additions are a single `register()` call when needed.

#### Metadata Contract

Every `Document` returned by any loader **must** include:

```
document.metadata = {
  "source":    "/absolute/path/to/file.pdf",
  "file_type": "pdf",
  "page":      3,          # where applicable (PDF only)
  "row":       12,         # where applicable (CSV only)
}
```

This metadata propagates through chunking and embedding unchanged, enabling citation
and source attribution at query time.

#### Error Handling Policy

- **Unsupported file type:** log a warning and skip the file. Do not raise an exception.
- **Corrupt / unreadable file:** log the error with the file path and skip. Raise a
  `ValueError` only if zero documents were loaded from the entire directory.
- **Empty directory:** raise a descriptive `ValueError` — this is a misconfiguration, not
a recoverable runtime error.

#### Key Design Decisions


| Decision                                 | Rationale                                                 |
| ---------------------------------------- | --------------------------------------------------------- |
| Factory/registry over `if/elif`          | Open/closed principle — extend without modifying          |
| Abstract base class                      | Enforces the `load()` contract on all concrete loaders    |
| PyMuPDFLoader preferred over PyPDFLoader | Faster, better text extraction, handles more PDF variants |
| Recursive directory scan                 | Supports nested folder structures in `data/`              |


---

### 3.2 `chunk_strategies.py` — Chunking

**Single Responsibility:** Accept loaded documents and return a flat list of overlapping
text chunks, with metadata preserved.

#### Class Design

```
ChunkStrategy  (abstract base class)
├── __init__(chunk_size: int, chunk_overlap: int)
└── split(documents: List[Document]) → List[Document]   ← abstract

RecursiveCharacterChunkStrategy(ChunkStrategy)
└── split() → uses RecursiveCharacterTextSplitter

SemanticChunkStrategy(ChunkStrategy)   ← future, not in scope yet
└── split() → uses SemanticChunker
```

#### Why the Strategy Pattern Here

Even though only `RecursiveCharacterChunkStrategy` is in scope, the Strategy pattern
costs nothing upfront and eliminates a future refactor. Different document types may
warrant different chunking strategies (e.g., semantic chunking for narrative text,
fixed-size chunking for structured CSVs).

#### Configuration

```
default chunk_size:    1000  (characters)
default chunk_overlap: 200   (characters)
```

These are constructor-injected, not hardcoded. The pipeline passes them from a central
config object, making tuning a config change, not a code change.

#### Metadata Propagation

LangChain's `RecursiveCharacterTextSplitter` preserves `document.metadata` on each chunk
by default. The chunker must additionally inject:

```
chunk.metadata["chunk_index"] = 4       # position within source document
chunk.metadata["chunk_total"] = 22      # total chunks from that document
```

This supports ordered reconstruction and debugging.

#### Key Design Decisions


| Decision                         | Rationale                                                                          |
| -------------------------------- | ---------------------------------------------------------------------------------- |
| Strategy pattern                 | Enables future semantic/markdown-aware chunkers without pipeline changes           |
| Metadata injection               | Chunk index + total required for traceability and ordered retrieval                |
| `RecursiveCharacterTextSplitter` | Best general-purpose splitter; respects natural boundaries (paragraphs, sentences) |
| Overlap at 20% of chunk size     | Standard practice — preserves cross-boundary context                               |


---

### 3.3 `embedding_service.py` — Embedding

**Single Responsibility:** Convert a list of document chunks into a parallel list of
float vectors using a HuggingFace Sentence Transformer model.

#### Class Design

```
EmbeddingService
├── __init__(model_name: str, batch_size: int = 32)
├── _model: HuggingFaceEmbeddings       # loaded once at init, not per call
├── embed(chunks: List[Document]) → List[List[float]]
└── _embed_batch(batch: List[str]) → List[List[float]]   # internal
```

#### Batching Strategy

Embedding all chunks in one call is unsafe at production scale — a corpus of 10,000 chunks
will exhaust GPU/CPU memory. The service batches internally:

```
embed(chunks):
  texts = [chunk.page_content for chunk in chunks]
  results = []
  for i in range(0, len(texts), batch_size):
      batch = texts[i : i + batch_size]
      results.extend(_embed_batch(batch))
  return results
```

`batch_size` defaults to 32 but is constructor-injected for tuning.

#### Model Loading

The HuggingFace model is instantiated **once in `__init__`**, not on each `embed()` call.
This is critical — model loading downloads weights and takes several seconds. Loading it
lazily or per-call would make the pipeline unusable at scale.

Recommended default model: `sentence-transformers/all-MiniLM-L6-v2`

- 384-dimensional output
- Fast inference
- Strong general-purpose semantic similarity performance
- Always runs on **CPU** — no GPU detection or CUDA dependency

#### Output Contract

`embed()` returns `List[List[float]]` where:

- `len(output) == len(chunks)` — always, guaranteed
- `output[i]` is the embedding for `chunks[i]` — positional correspondence is preserved

The pipeline relies on this positional guarantee when pairing chunks with embeddings
for storage.

#### Key Design Decisions


| Decision                                     | Rationale                                                         |
| -------------------------------------------- | ----------------------------------------------------------------- |
| Model loaded at `__init__`                   | Avoid repeated cold-start latency; fail fast if model unavailable |
| Batch processing                             | Prevents OOM on large corpora                                     |
| Constructor-injected model name              | Swap models without touching pipeline code                        |
| Returns `List[List[float]]` not `np.ndarray` | Framework-agnostic; FAISS accepts both but list is more portable  |


---

### 3.4 `vectordb_factory.py` — Vector Storage

**Single Responsibility:** Accept chunks and their embeddings, write them to a FAISS
index in batches, and persist the index to disk.

#### Class Design

```
VectorStore  (abstract base class)
├── store(chunks: List[Document], embeddings: List[List[float]]) → None
└── persist() → None

FAISSVectorStore(VectorStore)
├── __init__(persist_path: str, batch_size: int = 500)
├── _index: faiss.Index | None
├── store(chunks, embeddings) → None
└── persist() → None   (calls FAISS.save_local())

VectorDBFactory
├── _registry: dict[str, type[VectorStore]]
├── register(db_type, store_class)
└── create(db_type: str, **kwargs) → VectorStore
```

#### Batch Write Strategy

Writing all embeddings at once is not safe for large corpora. The store method processes
in configurable batches:

```
store(chunks, embeddings):
  for i in range(0, len(chunks), batch_size):
      batch_chunks = chunks[i : i + batch_size]
      batch_embeddings = embeddings[i : i + batch_size]
      if self._index is None:
          self._index = FAISS.from_embeddings(batch_embeddings, batch_chunks)
      else:
          self._index.add_embeddings(batch_embeddings, batch_chunks)
  self.persist()
```

`persist()` is called once after all batches, not after each batch, to avoid
redundant I/O.

#### FAISS Persistence

The index is saved to disk via `save_local(persist_path)`. The persist path is
constructor-injected and should default to a `vectorstore/` directory inside the project.
Without `save_local()`, the index is lost when the process exits.

#### Factory Pattern Rationale

`VectorDBFactory` enables future backends (Chroma, Pinecone, Weaviate) to be added as
a new `register()` call and a new concrete class. The pipeline code never references
`FAISSVectorStore` directly — only `VectorDBFactory.create("faiss")`.

#### Key Design Decisions


| Decision                            | Rationale                                                 |
| ----------------------------------- | --------------------------------------------------------- |
| Factory pattern                     | Swap vector DB backend via config change, not code change |
| Abstract `VectorStore` base         | Enforces `store()` + `persist()` contract on all backends |
| Batch writes                        | Safe for large corpora; first batch bootstraps the index  |
| `persist()` called once post-loop   | Avoid repeated disk I/O; atomic write at the end          |
| `persist_path` constructor-injected | Configurable, no hardcoded paths                          |


---

### 3.5 `ingestion_pipeline.py` — Orchestration

**Single Responsibility:** Call each component in order and pass results between steps.
Zero business logic lives here — it is a pure coordinator.

#### Class Design

```
IngestionPipeline
├── __init__(
│     loader: DocumentLoaderFactory,
│     chunker: ChunkStrategy,
│     embedder: EmbeddingService,
│     vector_store: VectorStore,
│   )
└── run(data_dir: str) → None
```

All four components are **constructor-injected**. The pipeline never instantiates them
internally. This makes the pipeline fully testable — in unit tests, each component can
be replaced with a mock.

#### Execution Logic

```
run(data_dir):
  documents  = self.loader.load_all(data_dir)
  chunks     = self.chunker.split(documents)
  embeddings = self.embedder.embed(chunks)
  self.vector_store.store(chunks, embeddings)
```

Each line is a direct method call on an injected component. Results flow as plain
Python variables — no state objects, no graph nodes, no routing logic.

#### Key Design Decisions

| Decision                                 | Rationale                                                      |
| ---------------------------------------- | -------------------------------------------------------------- |
| Constructor injection for all components | Full testability; no hidden instantiation                      |
| Plain sequential method calls            | Simplest correct approach; easy to read, debug, and extend     |
| Single `run()` entry point               | Clean public API; caller needs no knowledge of internals       |
| No internal state object                 | Data passes as local variables — no shared mutable state risks |
| `try/except` at `run()` level            | Single point for error logging and re-raising with context     |


---

## 4. Design Patterns Used


| Pattern                     | Where Applied                                        | Why                                        |
| --------------------------- | ---------------------------------------------------- | ------------------------------------------ |
| **Abstract Base Class**     | `BaseDocumentLoader`, `ChunkStrategy`, `VectorStore` | Enforces contracts, enables polymorphism   |
| **Factory / Registry**      | `DocumentLoaderFactory`, `VectorDBFactory`           | Open/closed — extend without modifying     |
| **Strategy**                | `ChunkStrategy` and subclasses                       | Swap algorithms without changing pipeline  |
| **Dependency Injection**    | `IngestionPipeline.__init__`                         | Testability, decoupling, flexibility       |
| **Template Method**         | Abstract base `load()` / `split()` / `store()`       | Shared structure, variable implementation  |
| **Sequential Pipeline**     | `IngestionPipeline.run()` method call chain          | Simplest correct pattern; output of each step is input to the next |


---

## 5. Cross-Cutting Concerns

### Logging

- Use Python's standard `logging` module (not `print`).
- Log at the start and end of each pipeline step with document/chunk/embedding counts.
- Log file-level loader decisions (which loader was selected for which file).
- Log batch progress in the embedder and vector store.
- Errors are logged at `ERROR` level with full tracebacks before re-raising.

### Configuration

All tunable parameters live in a single `PipelineConfig` dataclass:

```
PipelineConfig
├── data_dir: str
├── chunk_size: int = 1000
├── chunk_overlap: int = 200
├── embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
├── embedding_batch_size: int = 32
├── vector_db_type: str = "faiss"
├── vector_db_persist_path: str = "vectorstore/"
└── vector_db_batch_size: int = 500
```

The pipeline reads from this config at construction time. Tuning the pipeline is a config
change, not a code change.

### Testability

- Each component is independently unit-testable with no external framework dependency.
- The pipeline is integration-testable by injecting mock components via the constructor.
- The factory's registry is testable by registering a mock loader class.

### Metadata Traceability

Metadata flows end-to-end:

```
File on disk
  → Document.metadata: {source, file_type, page}
      → Chunk.metadata: {source, file_type, page, chunk_index, chunk_total}
          → FAISS docstore: chunk stored with full metadata
              → Retrieved at query time with citation info intact
```

---

## 6. Dependency Map

### Python Package Dependencies (to be added to pyproject.toml)


| Package                 | Version Constraint | Purpose                                           |
| ----------------------- | ------------------ | ------------------------------------------------- |
| `langchain`             | `>=0.3`            | Core Document type, loaders, splitters            |
| `langchain-community`   | `>=0.3`            | PyPDFLoader, CSVLoader, TextLoader                |
| `langchain-huggingface` | `>=0.1`            | HuggingFaceEmbeddings                             |
| `faiss-cpu`             | `>=1.8`            | Vector storage (use `faiss-gpu` if GPU available) |
| `sentence-transformers` | `>=3.0`            | HuggingFace embedding models                      |
| `pymupdf`               | `>=1.24`           | PDF loading (PyMuPDFLoader)                       |
| `python-docx`           | `>=1.1`            | DOCX support (deferred — not in initial scope)    |
| `openpyxl`              | `>=3.1`            | XLSX support (deferred — not in initial scope)    |


### Internal Module Dependency Graph

```
ingestion_pipeline.py
  ├── base_loader.py
  ├── chunk_strategies.py
  ├── embedding_service.py
  └── vectordb_factory.py

base_loader.py        → no internal deps
chunk_strategies.py   → no internal deps
embedding_service.py  → no internal deps
vectordb_factory.py   → no internal deps
```

All four leaf modules are fully independent of each other. Only `ingestion_pipeline.py`
imports from them. This structure eliminates circular imports and makes each module
independently testable.

---

## 7. Risk Register


| Risk                                                          | Severity | Likelihood | Mitigation                                                                   |
| ------------------------------------------------------------- | -------- | ---------- | ---------------------------------------------------------------------------- |
| Memory exhaustion holding all documents + chunks in memory    | High     | Medium     | Streaming / lazy iteration for very large data dirs                          |
| FAISS index lost on process exit if `save_local()` not called | High     | Low        | Always call `persist()` in a `finally` block inside `store()`                |
| HuggingFace model download fails (no internet / firewall)     | Medium   | Medium     | Pre-download model; cache path configurable via `TRANSFORMERS_CACHE` env var |
| Non-UTF-8 encoded text files crash `TextLoader`               | Medium   | Medium     | Explicit `encoding="utf-8", errors="replace"` in TextLoader constructor      |
| Unsupported file type silently skipped → empty document list  | Medium   | Low        | Log warning per skipped file; raise if total documents == 0                  |
| FAISS not thread-safe for concurrent writes                   | Low      | Low        | Pipeline is single-threaded; note in docs if parallelism is added later      |
| Chunk/embedding count mismatch breaks positional guarantee    | High     | Low        | Assert `len(embeddings) == len(chunks)` before calling `store()`             |
| PyPDFLoader fails on scanned/image PDFs                       | Medium   | Medium     | Use PyMuPDFLoader as default; it handles more PDF variants                   |


---

## 8. Open Questions

These are unresolved design questions that should be answered before implementation begins:

1. ~~**Incremental ingestion:**~~ **Resolved — not needed.** Re-running the pipeline always
   replaces the existing FAISS index. No append or deduplication logic required.
2. **Document-level vs. chunk-level embeddings:** The current design embeds chunks only.
  Should document-level summary embeddings also be stored for coarse retrieval?
3. ~~**DOCX / XLSX / JSON loaders:**~~ **Resolved — deferred.** Initial implementation
   supports PDF, TXT, and CSV only. DOCX, XLSX, and JSON are out of scope for now.
4. ~~**GPU vs. CPU embedding:**~~ **Resolved — CPU only.** `EmbeddingService` always runs
   on CPU. No CUDA detection or GPU dependency.
5. **Persist path convention:** Should the FAISS index be saved relative to the project
  root, relative to `data/`, or to an absolute path from config?
6. **Pipeline re-entrancy:** If the pipeline fails mid-run (e.g., after embedding but
  before storing), should it resume from the last successful step or restart from scratch?
7. **Logging destination:** Should logs go to stdout only, or also to a rotating file
  handler? Is there a requirement to surface logs to an external system?

---