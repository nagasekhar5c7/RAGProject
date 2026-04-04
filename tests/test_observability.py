"""
test_observability.py — Tests for Observability + PipelineMetrics.

Tests step timing, metric recording, and quality score validation.
No API keys needed — LangSmith tracing is disabled in tests.
"""

import time

import pytest
from langchain_core.documents import Document
from observability import Observability, PipelineMetrics


class TestPipelineMetrics:
    """Tests for the PipelineMetrics dataclass."""

    def test_default_values(self):
        """All numeric fields default to 0, quality scores default to None."""
        m = PipelineMetrics(query="test")
        assert m.query == "test"
        assert m.rewritten_query == ""
        assert m.latency_total == 0.0
        assert m.chunks_retrieved == 0
        assert m.estimated_total_tokens == 0
        assert m.relevancy_score is None
        assert m.hallucination_rate is None

    def test_to_dict_returns_all_fields(self):
        """to_dict() contains all expected keys."""
        m = PipelineMetrics(query="test query")
        d = m.to_dict()

        expected_keys = [
            "query", "rewritten_query",
            "latency_query_understanding_s", "latency_retrieval_s",
            "latency_reranking_s", "latency_context_building_s",
            "latency_llm_s", "latency_total_s",
            "chunks_retrieved", "chunks_after_rerank",
            "avg_retrieval_score", "avg_rerank_score",
            "estimated_prompt_tokens", "estimated_completion_tokens",
            "estimated_total_tokens",
            "relevancy_score", "hallucination_rate",
            "answer_length",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_rounds_latencies(self):
        """Latency values are rounded to 3 decimal places."""
        m = PipelineMetrics(query="test")
        m.latency_retrieval = 1.23456789
        d = m.to_dict()
        assert d["latency_retrieval_s"] == 1.235


class TestStepTimer:
    """Tests for Observability.step_timer()."""

    def test_records_elapsed_time(self):
        """step_timer() records non-zero duration for a timed step."""
        obs = Observability()
        metrics = obs.start_run("test query")

        with obs.step_timer("retrieval", metrics):
            time.sleep(0.05)

        assert metrics.latency_retrieval >= 0.04  # allow some variance

    def test_all_step_names_work(self):
        """All 5 supported step names record their latency."""
        obs = Observability()
        metrics = obs.start_run("test")

        steps = ["query_understanding", "retrieval", "reranking", "context_building", "llm"]
        for step in steps:
            with obs.step_timer(step, metrics):
                time.sleep(0.01)

        assert metrics.latency_query_understanding > 0
        assert metrics.latency_retrieval > 0
        assert metrics.latency_reranking > 0
        assert metrics.latency_context_building > 0
        assert metrics.latency_llm > 0

    def test_invalid_step_name_raises(self):
        """An invalid step name raises ValueError."""
        obs = Observability()
        metrics = obs.start_run("test")

        with pytest.raises(ValueError, match="Unknown step"):
            with obs.step_timer("invalid_step", metrics):
                pass


class TestRecordRetrievalChunks:
    """Tests for Observability.record_retrieval_chunks()."""

    def test_records_count_and_avg_score(self):
        """Records chunk count and average similarity_score."""
        obs = Observability()
        metrics = obs.start_run("test")

        chunks = [
            Document(page_content="a", metadata={"similarity_score": 0.8}),
            Document(page_content="b", metadata={"similarity_score": 0.6}),
        ]
        obs.record_retrieval_chunks(chunks, metrics)

        assert metrics.chunks_retrieved == 2
        assert abs(metrics.avg_retrieval_score - 0.7) < 0.001

    def test_empty_chunks(self):
        """Empty chunk list sets count to 0 and score stays 0."""
        obs = Observability()
        metrics = obs.start_run("test")
        obs.record_retrieval_chunks([], metrics)

        assert metrics.chunks_retrieved == 0
        assert metrics.avg_retrieval_score == 0.0


class TestRecordRerankChunks:
    """Tests for Observability.record_rerank_chunks()."""

    def test_records_count_and_avg_score(self):
        """Records chunk count and average relevance_score."""
        obs = Observability()
        metrics = obs.start_run("test")

        chunks = [
            Document(page_content="a", metadata={"relevance_score": 0.9}),
            Document(page_content="b", metadata={"relevance_score": 0.7}),
            Document(page_content="c", metadata={"relevance_score": 0.5}),
        ]
        obs.record_rerank_chunks(chunks, metrics)

        assert metrics.chunks_after_rerank == 3
        assert abs(metrics.avg_rerank_score - 0.7) < 0.001


class TestRecordTokens:
    """Tests for Observability.record_tokens()."""

    def test_estimates_token_counts(self):
        """Token counts are estimated as len(text) // 4."""
        obs = Observability()
        metrics = obs.start_run("test")

        prompt = "a" * 400  # ~100 tokens
        answer = "b" * 200  # ~50 tokens
        obs.record_tokens(prompt, answer, metrics)

        assert metrics.estimated_prompt_tokens == 100
        assert metrics.estimated_completion_tokens == 50
        assert metrics.estimated_total_tokens == 150
        assert metrics.answer_length == 200


class TestSetQualityScores:
    """Tests for Observability.set_quality_scores()."""

    def test_sets_relevancy_score(self):
        """Relevancy score is set on the metrics object."""
        obs = Observability()
        metrics = obs.start_run("test")
        obs.set_quality_scores(metrics, relevancy_score=0.85)
        assert metrics.relevancy_score == 0.85

    def test_sets_hallucination_rate(self):
        """Hallucination rate is set on the metrics object."""
        obs = Observability()
        metrics = obs.start_run("test")
        obs.set_quality_scores(metrics, hallucination_rate=0.1)
        assert metrics.hallucination_rate == 0.1

    def test_invalid_relevancy_raises(self):
        """Relevancy score outside [0, 1] raises ValueError."""
        obs = Observability()
        metrics = obs.start_run("test")
        with pytest.raises(ValueError):
            obs.set_quality_scores(metrics, relevancy_score=1.5)

    def test_invalid_hallucination_raises(self):
        """Hallucination rate outside [0, 1] raises ValueError."""
        obs = Observability()
        metrics = obs.start_run("test")
        with pytest.raises(ValueError):
            obs.set_quality_scores(metrics, hallucination_rate=-0.1)

    def test_none_values_are_ignored(self):
        """Passing None leaves the existing value unchanged."""
        obs = Observability()
        metrics = obs.start_run("test")
        metrics.relevancy_score = 0.8
        obs.set_quality_scores(metrics, relevancy_score=None)
        assert metrics.relevancy_score == 0.8
