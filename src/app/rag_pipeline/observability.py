"""
observability.py — Step 7: Pipeline Observability & Tracing

Responsibility:
    Track per-step latency, token usage, retrieval quality scores, and
    optional quality metrics (relevancy score, hallucination rate) for
    every pipeline run.  Sends LLM traces to LangSmith automatically by
    setting the required environment variables at import time.

Two components:
    PipelineMetrics  — dataclass that holds all measurable values for one run.
    Observability    — manages LangSmith config, provides a step_timer()
                       context manager, and logs a structured summary at the
                       end of each run.

Metrics collected (Phase 1 — current):
    Latency
        latency_query_understanding  (seconds)
        latency_retrieval            (seconds)
        latency_reranking            (seconds)
        latency_context_building     (seconds)
        latency_llm                  (seconds)
        latency_total                (seconds)
    Tokens (estimated: len(text) // 4)
        estimated_prompt_tokens
        estimated_completion_tokens
        estimated_total_tokens
    Retrieval quality
        chunks_retrieved             (count from FAISS)
        chunks_after_rerank          (count after Cohere rerank)
        avg_retrieval_score          (mean similarity_score from retriever)
        avg_rerank_score             (mean relevance_score from reranker)

Metrics collected (Phase 2 — pluggable, set externally):
    relevancy_score     (float 0-1) — how relevant is the answer to the query
    hallucination_rate  (float 0-1) — fraction of answer not grounded in context

LangSmith:
    Set LANGSMITH_API_KEY and LANGSMITH_TRACING_ENABLED = True in config.py.
    All LangChain calls (ChatGroq, HuggingFaceEmbeddings, CohereRerank, FAISS)
    are traced automatically — no code changes needed in other pipeline files.

Config:
    LANGSMITH_API_KEY         = ""          (LangSmith API key)
    LANGSMITH_PROJECT         = "BookRAG"  (project name in LangSmith UI)
    LANGSMITH_TRACING_ENABLED = True        (set False to disable)
"""

import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
from config.config import (
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING_ENABLED,
)

from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Pipeline Metrics Dataclass
# ──────────────────────────────────────────────

@dataclass
class PipelineMetrics:
    """
    Holds all measurable values captured during a single RAG pipeline run.

    Latency fields are populated by the Observability.step_timer() context
    manager.  Quality fields (relevancy_score, hallucination_rate) are
    optional and must be set externally after answer evaluation.

    Attributes:
        query (str):                        Raw user query that started the run.
        rewritten_query (str):              Query after QueryUnderstanding.rewrite().

        latency_query_understanding (float): Seconds spent in Step 1.
        latency_retrieval (float):           Seconds spent in Step 2.
        latency_reranking (float):           Seconds spent in Step 3.
        latency_context_building (float):    Seconds spent in Step 4.
        latency_llm (float):                 Seconds spent in Step 5 (LLM call).
        latency_total (float):               Wall-clock time for the full run.

        chunks_retrieved (int):              Chunks returned by the retriever.
        chunks_after_rerank (int):           Chunks returned by the reranker.
        avg_retrieval_score (float):         Mean similarity_score across retrieved chunks.
        avg_rerank_score (float):            Mean relevance_score across reranked chunks.

        estimated_prompt_tokens (int):       Estimated tokens in the system + user message.
        estimated_completion_tokens (int):   Estimated tokens in the LLM answer.
        estimated_total_tokens (int):        Sum of prompt + completion tokens.

        relevancy_score (float | None):      0-1 score: how relevant is the answer.
                                             None until evaluated externally.
        hallucination_rate (float | None):   0-1 rate: fraction of answer not in context.
                                             None until evaluated externally.

        answer_length (int):                 Character count of the final answer.
    """

    query: str

    # Step outputs
    rewritten_query: str = ""

    # Per-step latencies (seconds)
    latency_query_understanding: float = 0.0
    latency_retrieval:           float = 0.0
    latency_reranking:           float = 0.0
    latency_context_building:    float = 0.0
    latency_llm:                 float = 0.0
    latency_total:               float = 0.0

    # Retrieval quality
    chunks_retrieved:     int   = 0
    chunks_after_rerank:  int   = 0
    avg_retrieval_score:  float = 0.0
    avg_rerank_score:     float = 0.0

    # Token usage (estimated)
    estimated_prompt_tokens:     int = 0
    estimated_completion_tokens: int = 0
    estimated_total_tokens:      int = 0

    # Phase 2 quality scores — set externally after evaluation
    relevancy_score:    float | None = None
    hallucination_rate: float | None = None

    # Answer
    answer_length: int = 0

    def to_dict(self) -> dict:
        """
        Serialise the metrics to a plain dictionary (useful for logging / storage).

        Returns:
            dict: All metric fields as key-value pairs.
        """
        return {
            "query":                        self.query,
            "rewritten_query":              self.rewritten_query,
            "latency_query_understanding_s": round(self.latency_query_understanding, 3),
            "latency_retrieval_s":           round(self.latency_retrieval, 3),
            "latency_reranking_s":           round(self.latency_reranking, 3),
            "latency_context_building_s":    round(self.latency_context_building, 3),
            "latency_llm_s":                 round(self.latency_llm, 3),
            "latency_total_s":               round(self.latency_total, 3),
            "chunks_retrieved":              self.chunks_retrieved,
            "chunks_after_rerank":           self.chunks_after_rerank,
            "avg_retrieval_score":           round(self.avg_retrieval_score, 4),
            "avg_rerank_score":              round(self.avg_rerank_score, 4),
            "estimated_prompt_tokens":       self.estimated_prompt_tokens,
            "estimated_completion_tokens":   self.estimated_completion_tokens,
            "estimated_total_tokens":        self.estimated_total_tokens,
            "relevancy_score":               self.relevancy_score,
            "hallucination_rate":            self.hallucination_rate,
            "answer_length":                 self.answer_length,
        }


# ──────────────────────────────────────────────
# Observability
# ──────────────────────────────────────────────

class Observability:
    """
    Manages LangSmith tracing configuration and per-run metrics collection.

    LangSmith is activated by setting three environment variables at
    construction time.  Once set, every LangChain call in the process
    (ChatGroq, HuggingFaceEmbeddings, CohereRerank, FAISS) is traced
    automatically — no instrumentation needed in other files.

    Usage pattern in the orchestrator:
        obs = Observability()

        metrics = obs.start_run(query)

        with obs.step_timer("retrieval", metrics):
            chunks = retriever.retrieve(rewritten_query)

        obs.record_retrieval_chunks(chunks, metrics)
        obs.record_tokens(prompt_text, answer_text, metrics)
        obs.finish_run(metrics)

    Attributes:
        tracing_enabled (bool): Whether LangSmith tracing is active.
        project (str):          LangSmith project name.
    """

    # Step name → PipelineMetrics attribute that stores its duration
    _STEP_LATENCY_FIELD: dict[str, str] = {
        "query_understanding": "latency_query_understanding",
        "retrieval":           "latency_retrieval",
        "reranking":           "latency_reranking",
        "context_building":    "latency_context_building",
        "llm":                 "latency_llm",
    }

    def __init__(self):
        """
        Configure LangSmith tracing by setting the required environment
        variables from config.py values.

        LangSmith reads these env vars automatically:
            LANGCHAIN_TRACING_V2  = "true"
            LANGCHAIN_API_KEY     = <your key>
            LANGCHAIN_PROJECT     = <project name>

        If LANGSMITH_TRACING_ENABLED is False or the API key is empty,
        tracing is skipped and a warning is logged.
        """
        self.project = LANGSMITH_PROJECT
        self.tracing_enabled = LANGSMITH_TRACING_ENABLED and bool(LANGSMITH_API_KEY)

        if self.tracing_enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"]    = LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"]    = LANGSMITH_PROJECT
            logger.info(
                f"LangSmith tracing enabled — project='{LANGSMITH_PROJECT}'"
            )
        else:
            reason = "disabled in config" if not LANGSMITH_TRACING_ENABLED else "API key not set"
            logger.warning(
                f"LangSmith tracing is OFF ({reason}). "
                "Set LANGSMITH_API_KEY and LANGSMITH_TRACING_ENABLED=True in config.py to enable."
            )

    # ── Run lifecycle ─────────────────────────────────────────

    def start_run(self, query: str) -> PipelineMetrics:
        """
        Begin a new pipeline run and return a fresh PipelineMetrics object.

        Call this once at the start of RAGPipeline.run(), before any steps.

        Args:
            query (str): The raw user query for this run.

        Returns:
            PipelineMetrics: Initialised metrics object for this run.
        """
        logger.info(f"Observability: starting run for query '{query[:60]}'")
        return PipelineMetrics(query=query)

    def finish_run(self, metrics: PipelineMetrics) -> None:
        """
        Log a structured summary of all collected metrics for the run.

        Call this once at the end of RAGPipeline.run(), after the LLM step.

        Args:
            metrics (PipelineMetrics): The populated metrics object for this run.
        """
        logger.info(
            "\n"
            "┌─────────────────────────────────────────────────────┐\n"
            "│                  PIPELINE RUN SUMMARY                │\n"
            "└─────────────────────────────────────────────────────┘\n"
            f"  Query              : {metrics.query[:70]}\n"
            f"  Rewritten query    : {metrics.rewritten_query[:70]}\n"
            "  ── Latency ─────────────────────────────────────────\n"
            f"  Query understanding: {metrics.latency_query_understanding:.3f}s\n"
            f"  Retrieval          : {metrics.latency_retrieval:.3f}s\n"
            f"  Reranking          : {metrics.latency_reranking:.3f}s\n"
            f"  Context building   : {metrics.latency_context_building:.3f}s\n"
            f"  LLM call           : {metrics.latency_llm:.3f}s\n"
            f"  Total              : {metrics.latency_total:.3f}s\n"
            "  ── Retrieval ────────────────────────────────────────\n"
            f"  Chunks retrieved   : {metrics.chunks_retrieved}\n"
            f"  Chunks after rerank: {metrics.chunks_after_rerank}\n"
            f"  Avg retrieval score: {metrics.avg_retrieval_score:.4f}\n"
            f"  Avg rerank score   : {metrics.avg_rerank_score:.4f}\n"
            "  ── Tokens (estimated) ───────────────────────────────\n"
            f"  Prompt tokens      : ~{metrics.estimated_prompt_tokens}\n"
            f"  Completion tokens  : ~{metrics.estimated_completion_tokens}\n"
            f"  Total tokens       : ~{metrics.estimated_total_tokens}\n"
            "  ── Quality (Phase 2) ────────────────────────────────\n"
            f"  Relevancy score    : {metrics.relevancy_score if metrics.relevancy_score is not None else 'not evaluated'}\n"
            f"  Hallucination rate : {metrics.hallucination_rate if metrics.hallucination_rate is not None else 'not evaluated'}\n"
            f"  Answer length      : {metrics.answer_length} chars\n"
        )

    # ── Step timer ────────────────────────────────────────────

    @contextmanager
    def step_timer(
        self,
        step_name: str,
        metrics: PipelineMetrics,
    ) -> Generator[None, None, None]:
        """
        Context manager that measures wall-clock time for a pipeline step
        and stores it in the corresponding PipelineMetrics field.

        Usage:
            with obs.step_timer("retrieval", metrics):
                chunks = retriever.retrieve(query)

        Supported step names:
            "query_understanding", "retrieval", "reranking",
            "context_building", "llm"

        Args:
            step_name (str):         One of the supported step name strings.
            metrics (PipelineMetrics): The current run's metrics object.

        Yields:
            None

        Raises:
            ValueError: If step_name is not a recognised step.
        """
        if step_name not in self._STEP_LATENCY_FIELD:
            raise ValueError(
                f"Unknown step '{step_name}'. "
                f"Valid steps: {list(self._STEP_LATENCY_FIELD.keys())}"
            )

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            field_name = self._STEP_LATENCY_FIELD[step_name]
            setattr(metrics, field_name, elapsed)
            logger.debug(f"  [{step_name}] {elapsed:.3f}s")

    # ── Metric recorders ──────────────────────────────────────

    def record_retrieval_chunks(
        self,
        chunks: list[Document],
        metrics: PipelineMetrics,
    ) -> None:
        """
        Record retrieval quality metrics from the chunks returned by the Retriever.

        Computes:
            - chunks_retrieved      : total count
            - avg_retrieval_score   : mean of metadata["similarity_score"] across chunks

        Args:
            chunks  (list[Document]):  Chunks from retriever.retrieve().
            metrics (PipelineMetrics): The current run's metrics object.
        """
        metrics.chunks_retrieved = len(chunks)
        if chunks:
            scores = [
                doc.metadata.get("similarity_score", 0.0)
                for doc in chunks
            ]
            metrics.avg_retrieval_score = sum(scores) / len(scores)

    def record_rerank_chunks(
        self,
        chunks: list[Document],
        metrics: PipelineMetrics,
    ) -> None:
        """
        Record reranking quality metrics from the chunks returned by the Reranker.

        Computes:
            - chunks_after_rerank   : total count
            - avg_rerank_score      : mean of metadata["relevance_score"] across chunks

        Args:
            chunks  (list[Document]):  Chunks from reranker.rerank().
            metrics (PipelineMetrics): The current run's metrics object.
        """
        metrics.chunks_after_rerank = len(chunks)
        if chunks:
            scores = [
                doc.metadata.get("relevance_score", 0.0)
                for doc in chunks
            ]
            metrics.avg_rerank_score = sum(scores) / len(scores)

    def record_tokens(
        self,
        prompt_text: str,
        answer_text: str,
        metrics: PipelineMetrics,
    ) -> None:
        """
        Estimate and record token counts from the prompt and answer texts.

        Token estimation: len(text) // 4  (~4 characters per token for English).

        Args:
            prompt_text (str):         The full system + user message text sent to the LLM.
            answer_text (str):         The LLM-generated answer.
            metrics (PipelineMetrics): The current run's metrics object.
        """
        prompt_tokens     = max(0, len(prompt_text) // 4)
        completion_tokens = max(0, len(answer_text) // 4)

        metrics.estimated_prompt_tokens     = prompt_tokens
        metrics.estimated_completion_tokens = completion_tokens
        metrics.estimated_total_tokens      = prompt_tokens + completion_tokens
        metrics.answer_length               = len(answer_text)

    def set_quality_scores(
        self,
        metrics: PipelineMetrics,
        relevancy_score: float | None = None,
        hallucination_rate: float | None = None,
    ) -> None:
        """
        Set Phase 2 quality scores on the metrics object.

        These scores are not computed automatically — they must be provided
        by an external evaluation component (e.g. an LLM-based judge or
        RAGAS evaluator) and then passed here.

        Args:
            metrics (PipelineMetrics):         The current run's metrics object.
            relevancy_score (float | None):    0.0–1.0; how relevant is the answer.
            hallucination_rate (float | None): 0.0–1.0; fraction of answer not
                                               grounded in the retrieved context.
        """
        if relevancy_score is not None:
            if not (0.0 <= relevancy_score <= 1.0):
                raise ValueError(f"relevancy_score must be in [0, 1], got {relevancy_score}")
            metrics.relevancy_score = relevancy_score

        if hallucination_rate is not None:
            if not (0.0 <= hallucination_rate <= 1.0):
                raise ValueError(f"hallucination_rate must be in [0, 1], got {hallucination_rate}")
            metrics.hallucination_rate = hallucination_rate


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test — simulates a pipeline run using dummy data.

    Run from the project root:
        uv run python src/app/rag_pipeline/observability.py
    """
    from langchain_core.documents import Document

    obs = Observability()

    # Simulate a run
    metrics = obs.start_run("what is the attention mechanism in transformers?")
    metrics.rewritten_query = "attention mechanism in transformer models"

    # Simulate step timings
    with obs.step_timer("query_understanding", metrics):
        time.sleep(0.05)

    with obs.step_timer("retrieval", metrics):
        time.sleep(0.12)

    # Simulate chunk recording
    dummy_retrieval_chunks = [
        Document(page_content="chunk A", metadata={"similarity_score": 0.82}),
        Document(page_content="chunk B", metadata={"similarity_score": 0.74}),
        Document(page_content="chunk C", metadata={"similarity_score": 0.68}),
    ]
    obs.record_retrieval_chunks(dummy_retrieval_chunks, metrics)

    with obs.step_timer("reranking", metrics):
        time.sleep(0.08)

    dummy_rerank_chunks = [
        Document(page_content="chunk A", metadata={"relevance_score": 0.91}),
        Document(page_content="chunk C", metadata={"relevance_score": 0.77}),
    ]
    obs.record_rerank_chunks(dummy_rerank_chunks, metrics)

    with obs.step_timer("context_building", metrics):
        time.sleep(0.02)

    with obs.step_timer("llm", metrics):
        time.sleep(0.35)

    dummy_prompt = "You are an assistant. Context: ...\nQuestion: what is attention?"
    dummy_answer = "The attention mechanism allows the model to focus on relevant parts of the input."
    obs.record_tokens(dummy_prompt, dummy_answer, metrics)

    # Simulate total latency
    metrics.latency_total = sum([
        metrics.latency_query_understanding,
        metrics.latency_retrieval,
        metrics.latency_reranking,
        metrics.latency_context_building,
        metrics.latency_llm,
    ])

    # Phase 2 quality scores (simulated)
    obs.set_quality_scores(metrics, relevancy_score=0.88, hallucination_rate=0.05)

    obs.finish_run(metrics)

    print("\nMetrics dict:")
    for k, v in metrics.to_dict().items():
        print(f"  {k:<35}: {v}")
