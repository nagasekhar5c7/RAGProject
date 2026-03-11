"""
rag_pipeline_orc.py — RAG Query Pipeline Orchestrator

Responsibility:
    Wire all RAG pipeline steps together and expose a single entry point:
        RAGPipeline.run(query) -> str

Pipeline steps (in order):
    1. QueryUnderstanding  — clean + rewrite the raw query
    2. Retriever           — fetch top-k candidate chunks from FAISS
    3. Reranker            — rerank candidates, keep top-n (Cohere)
    4. ContextBuilder      — deduplicate, trim to token budget, format
    5. PromptBuilder       — assemble messages and call LLM for the answer
    6. Observability       — per-step latency, token usage, retrieval scores,
                             LangSmith tracing

Input  : raw user query (str)
Output : answer (str) — LLM-generated response grounded in the retrieved context

Design:
    All components are constructed once inside RAGPipeline.__init__ and
    reused across calls, so the FAISS index and embedding model are loaded
    only once per process.

    The single public method run(query) orchestrates the full pipeline and
    returns the final answer string.
"""

import logging
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

from context_builder import ContextBuilder
from llm_service import LLMService
from observability import Observability
from prompt_builder import PromptBuilder, SYSTEM_PROMPT_TEMPLATE
from query_understanding import QueryUnderstanding
from reranker import Reranker
from retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# RAG Pipeline Orchestrator
# ──────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG query pipeline orchestrator.

    Constructs and holds all pipeline components.  Each component is
    initialised once at construction time so expensive resources (FAISS
    index, embedding model, Groq client, Cohere client) are loaded only
    once per process.

    Usage:
        pipeline = RAGPipeline()
        answer   = pipeline.run("what is the attention mechanism?")

    Attributes:
        query_understanding (QueryUnderstanding): Cleans and rewrites queries.
        retriever           (Retriever):          Fetches candidates from FAISS.
        reranker            (Reranker):           Reranks candidates via Cohere.
        context_builder     (ContextBuilder):     Deduplicates and formats context.
        prompt_builder      (PromptBuilder):      Builds prompt and calls the LLM.
        obs                 (Observability):      Latency tracking + LangSmith tracing.
    """

    def __init__(self):
        """
        Initialise all pipeline components.

        Component wiring:
            LLMService           → QueryUnderstanding (for query rewriting)
            LLMService           → PromptBuilder      (for answer generation)
            Retriever            → loads FAISS index from disk
            Reranker             → connects to Cohere API
            ContextBuilder       → stateless, uses config defaults
            PromptBuilder        → wraps LLMService
            Observability        → configures LangSmith, tracks metrics

        The same LLMService instance is shared between QueryUnderstanding and
        PromptBuilder to avoid creating two separate Groq clients.

        Raises:
            RuntimeError: If any component fails to initialise (e.g. FAISS
                          index missing, invalid API key).
        """
        logger.info("Initialising RAG pipeline components...")

        # Observability first — sets LangSmith env vars before any LangChain
        # objects are constructed so all downstream calls are traced.
        self.obs = Observability()

        llm = LLMService()

        self.query_understanding = QueryUnderstanding(llm=llm)
        self.retriever           = Retriever()
        self.reranker            = Reranker()
        self.context_builder     = ContextBuilder()
        self.prompt_builder      = PromptBuilder(llm=llm)

        logger.info("RAG pipeline ready.")

    def run(self, query: str) -> str:
        """
        Execute the full RAG pipeline for a single user query.

        Steps:
            1. **QueryUnderstanding** — clean + LLM-rewrite the raw query.
            2. **Retriever**          — fetch top-k candidate chunks from FAISS
                                        using the rewritten query.
            3. **Reranker**           — rerank candidates with Cohere, keep top-n.
            4. **ContextBuilder**     — deduplicate chunks, enforce token budget,
                                        format into a context block string.
            5. **PromptBuilder**      — inject context into system prompt, call
                                        LLM, return the generated answer.

        Args:
            query (str): Raw user question (any casing / whitespace).

        Returns:
            str: LLM-generated answer grounded in the retrieved context.
                 If no relevant chunks are found, returns a "no results" message.

        Raises:
            ValueError:   If the query is empty or blank.
            RuntimeError: If any pipeline step fails.
        """
        if not query.strip():
            raise ValueError("Query is empty — cannot run the pipeline.")

        pipeline_start = time.perf_counter()
        metrics = self.obs.start_run(query)

        logger.info(f"\n{'='*60}")
        logger.info(f"RAG pipeline started for query: '{query[:80]}'")
        logger.info(f"{'='*60}")

        # ── Step 1: Query Understanding ───────────────────────
        logger.info("[Step 1/5] Query Understanding...")
        with self.obs.step_timer("query_understanding", metrics):
            rewritten_query = self.query_understanding.rewrite(query)
        metrics.rewritten_query = rewritten_query
        logger.info(f"  Raw      : {query}")
        logger.info(f"  Rewritten: {rewritten_query}")

        # ── Step 2: Retrieval ─────────────────────────────────
        logger.info("[Step 2/5] Retrieval...")
        with self.obs.step_timer("retrieval", metrics):
            chunks = self.retriever.retrieve(rewritten_query)
        self.obs.record_retrieval_chunks(chunks, metrics)
        logger.info(f"  Chunks retrieved: {len(chunks)}")

        if not chunks:
            logger.warning("No chunks retrieved. Returning no-results message.")
            metrics.latency_total = time.perf_counter() - pipeline_start
            self.obs.finish_run(metrics)
            return (
                "I could not find any relevant content in the knowledge base "
                "for your query. Please try rephrasing your question."
            )

        # ── Step 3: Reranking ─────────────────────────────────
        logger.info("[Step 3/5] Reranking...")
        with self.obs.step_timer("reranking", metrics):
            top_chunks = self.reranker.rerank(rewritten_query, chunks)
        self.obs.record_rerank_chunks(top_chunks, metrics)
        logger.info(f"  Chunks after reranking: {len(top_chunks)}")

        # ── Step 4: Context Building ──────────────────────────
        logger.info("[Step 4/5] Building context...")
        with self.obs.step_timer("context_building", metrics):
            context = self.context_builder.build(top_chunks)
        logger.info(f"  Context size: ~{len(context) // 4} tokens")

        if not context.strip():
            logger.warning("Context is empty after building. Returning no-results message.")
            metrics.latency_total = time.perf_counter() - pipeline_start
            self.obs.finish_run(metrics)
            return (
                "I could not find a sufficient answer in the provided context. "
                "The retrieved chunks were empty or too short after processing."
            )

        # ── Step 5: Prompt + LLM ──────────────────────────────
        logger.info("[Step 5/5] Generating answer...")
        with self.obs.step_timer("llm", metrics):
            answer = self.prompt_builder.generate(
                query=rewritten_query,
                context=context,
            )

        # Record token usage using the full prompt text for estimation
        prompt_text = SYSTEM_PROMPT_TEMPLATE.format(context=context) + "\n" + rewritten_query
        self.obs.record_tokens(prompt_text, answer, metrics)

        metrics.latency_total = time.perf_counter() - pipeline_start
        self.obs.finish_run(metrics)

        logger.info(f"Pipeline complete. Answer length: {len(answer)} character(s).")
        logger.info(f"{'='*60}\n")

        return answer


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Interactive CLI for the RAG pipeline.

    Initialises the pipeline once, then accepts user queries in a loop
    until the user types 'exit' or 'quit'.

    Requires:
        - Built FAISS index at vectorstore/faiss_index/ (run ingestion first)
        - Valid GROQ_API_KEY and COHERE_API_KEY in config/config.py

    Run from the project root:
        uv run python src/app/rag_pipeline/rag_pipeline_orc.py
    """
    print("\n" + "=" * 60)
    print("  BookRAG — Retrieval-Augmented Generation Pipeline")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    pipeline = RAGPipeline()

    while True:
        try:
            user_query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_query:
            print("  Please enter a question.\n")
            continue

        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        answer = pipeline.run(user_query)

        print(f"\n{'─'*60}")
        print(f"Answer:\n{answer}")
        print(f"{'─'*60}\n")
