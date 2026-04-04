"""
test_api.py — Tests for the FastAPI application (POST /ask, GET /health).

Uses FastAPI's TestClient and mocks RAGPipeline — no FAISS, no API keys needed.
The lifespan is replaced with a no-op so the real pipeline never loads.
"""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_module


# ── No-op lifespan (skip loading real pipeline) ──────────────

@asynccontextmanager
async def _noop_lifespan(app):
    """Replaces the real lifespan so no FAISS/embeddings are loaded."""
    yield


# Swap the lifespan before any TestClient is created
api_module.app.router.lifespan_context = _noop_lifespan


@pytest.fixture
def client():
    """
    Create a TestClient with a mocked RAGPipeline.

    Sets api_module._pipeline to a MagicMock so the /ask endpoint
    returns a canned answer without any real pipeline call.
    """
    mock_instance = MagicMock()
    mock_instance.run.return_value = "The attention mechanism allows the model to focus on relevant parts."

    original = api_module._pipeline
    api_module._pipeline = mock_instance

    with TestClient(api_module.app) as c:
        yield c

    api_module._pipeline = original


@pytest.fixture
def client_no_pipeline():
    """
    Create a TestClient with _pipeline = None (simulates startup not finished).
    """
    original = api_module._pipeline
    api_module._pipeline = None

    with TestClient(api_module.app) as c:
        yield c

    api_module._pipeline = original


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 when pipeline is ready."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_fields(self, client):
        """Health response contains status, pipeline_ready, version."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["pipeline_ready"] is True
        assert "version" in data

    def test_health_has_request_id_header(self, client):
        """Health response includes X-Request-ID header."""
        response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_health_has_process_time_header(self, client):
        """Health response includes X-Process-Time-Ms header."""
        response = client.get("/health")
        assert "x-process-time-ms" in response.headers


class TestAskEndpoint:
    """Tests for POST /ask."""

    def test_ask_returns_200(self, client):
        """Valid query returns 200 with an answer."""
        response = client.post("/ask", json={"query": "what is attention?"})
        assert response.status_code == 200

    def test_ask_response_fields(self, client):
        """Response contains request_id, query, answer, latency_ms."""
        response = client.post("/ask", json={"query": "what is attention?"})
        data = response.json()
        assert "request_id" in data
        assert data["query"] == "what is attention?"
        assert "answer" in data
        assert "latency_ms" in data

    def test_ask_returns_pipeline_answer(self, client):
        """The answer field comes from the RAGPipeline.run() mock."""
        response = client.post("/ask", json={"query": "what is attention?"})
        data = response.json()
        assert "attention mechanism" in data["answer"]

    def test_ask_has_request_id_header(self, client):
        """Ask response includes X-Request-ID header."""
        response = client.post("/ask", json={"query": "what is attention?"})
        assert "x-request-id" in response.headers

    def test_ask_latency_is_positive(self, client):
        """latency_ms is a positive number."""
        response = client.post("/ask", json={"query": "what is attention?"})
        data = response.json()
        assert data["latency_ms"] > 0


class TestAskValidation:
    """Tests for POST /ask input validation."""

    def test_empty_body_returns_422(self, client):
        """Missing query field returns 422."""
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_query_too_short_returns_422(self, client):
        """Query shorter than 3 characters returns 422."""
        response = client.post("/ask", json={"query": "ab"})
        assert response.status_code == 422

    def test_query_too_long_returns_422(self, client):
        """Query longer than 500 characters returns 422."""
        response = client.post("/ask", json={"query": "a" * 501})
        assert response.status_code == 422

    def test_no_json_body_returns_422(self, client):
        """Request without JSON body returns 422."""
        response = client.post("/ask")
        assert response.status_code == 422


class TestPipelineNotReady:
    """Tests for when pipeline is not yet initialized."""

    def test_ask_returns_503_when_pipeline_not_ready(self, client_no_pipeline):
        """POST /ask returns 503 if pipeline hasn't loaded yet."""
        response = client_no_pipeline.post("/ask", json={"query": "what is attention?"})
        assert response.status_code == 503
