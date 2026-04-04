"""
test_context_builder.py — Tests for ContextBuilder.

Tests the 5-stage pipeline: clean → deduplicate → order → trim → format.
Uses sample Document fixtures — no API keys or FAISS needed.
"""

import pytest
from langchain_core.documents import Document
from context_builder import ContextBuilder


class TestCleanStage:
    """Tests for the text cleaning stage."""

    def test_collapses_whitespace_in_chunks(self, sample_chunks):
        """Extra spaces inside chunk content are collapsed."""
        chunk = Document(
            page_content="hello    world   test",
            metadata={"relevance_score": 0.9},
        )
        builder = ContextBuilder(max_tokens=5000)
        result = builder.build([chunk])
        assert "hello world test" in result

    def test_removes_excessive_newlines(self):
        """Runs of 3+ newlines are collapsed to 2."""
        chunk = Document(
            page_content="paragraph one\n\n\n\n\nparagraph two",
            metadata={"relevance_score": 0.9},
        )
        builder = ContextBuilder(max_tokens=5000)
        result = builder.build([chunk])
        assert "\n\n\n" not in result
        assert "paragraph one\n\nparagraph two" in result


class TestDeduplication:
    """Tests for the SHA-256 fingerprint deduplication stage."""

    def test_removes_exact_duplicates(self, duplicate_chunks):
        """Identical content from different sources is deduplicated."""
        builder = ContextBuilder(max_tokens=5000)
        result = builder.build(duplicate_chunks)

        # "focus on relevant parts" appears in 2 chunks but should only appear once
        count = result.count("focus on relevant parts")
        assert count == 1

    def test_keeps_unique_chunks(self, duplicate_chunks):
        """Non-duplicate chunks survive deduplication."""
        builder = ContextBuilder(max_tokens=5000)
        result = builder.build(duplicate_chunks)
        assert "self-attention" in result

    def test_first_occurrence_wins(self, duplicate_chunks):
        """The first occurrence (higher relevance_score) is kept."""
        builder = ContextBuilder(max_tokens=5000)
        result = builder.build(duplicate_chunks)
        # The first chunk (book_a.pdf, score 0.90) should win over book_b.pdf (0.85)
        assert "book_a.pdf" in result

    def test_no_duplicates_in_unique_list(self, sample_chunks):
        """All 5 unique chunks survive deduplication."""
        builder = ContextBuilder(max_tokens=50000)
        result = builder.build(sample_chunks)
        assert "Chunk 5" in result


class TestOrdering:
    """Tests for the ordering stage (sort by relevance_score desc)."""

    def test_chunks_ordered_by_relevance_desc(self, sample_chunks):
        """Chunk 1 in the output should have the highest relevance score."""
        builder = ContextBuilder(max_tokens=50000)
        result = builder.build(sample_chunks)

        # Find positions — Chunk 1 should appear before Chunk 5
        pos_chunk1 = result.find("Chunk 1")
        pos_chunk5 = result.find("Chunk 5")
        assert pos_chunk1 < pos_chunk5


class TestTokenBudget:
    """Tests for the token trimming stage."""

    def test_respects_max_tokens(self):
        """Context output stays within the token budget."""
        chunks = [
            Document(
                page_content="a" * 400,  # ~100 tokens
                metadata={"relevance_score": 0.9},
            ),
            Document(
                page_content="b" * 400,  # ~100 tokens
                metadata={"relevance_score": 0.8},
            ),
            Document(
                page_content="c" * 400,  # ~100 tokens
                metadata={"relevance_score": 0.7},
            ),
        ]
        # Budget of 150 tokens — should fit first chunk, maybe second, not all 3
        builder = ContextBuilder(max_tokens=150)
        result = builder.build(chunks)

        estimated_tokens = len(result) // 4
        assert estimated_tokens <= 200  # some overhead from headers

    def test_zero_budget_returns_empty(self):
        """A max_tokens of 0 returns empty string (no chunks fit)."""
        chunk = Document(page_content="hello world", metadata={"relevance_score": 0.9})
        builder = ContextBuilder(max_tokens=0)
        result = builder.build([chunk])
        assert result == ""

    def test_large_budget_keeps_all_chunks(self, sample_chunks):
        """A very large budget keeps all chunks."""
        builder = ContextBuilder(max_tokens=100000)
        result = builder.build(sample_chunks)
        assert "Chunk 5" in result


class TestFormatting:
    """Tests for the final formatting stage."""

    def test_contains_chunk_headers(self, sample_chunks):
        """Each chunk in the output has a [Chunk N | ...] header."""
        builder = ContextBuilder(max_tokens=50000)
        result = builder.build(sample_chunks)
        assert "[Chunk 1" in result
        assert "[Chunk 2" in result

    def test_contains_source_in_header(self, sample_chunks):
        """Source filename appears in the chunk header."""
        builder = ContextBuilder(max_tokens=50000)
        result = builder.build(sample_chunks)
        assert "Source: transformers.pdf" in result

    def test_contains_page_in_header(self, sample_chunks):
        """Page number appears in the chunk header."""
        builder = ContextBuilder(max_tokens=50000)
        result = builder.build(sample_chunks)
        assert "Page: 12" in result

    def test_separator_between_chunks(self, sample_chunks):
        """Chunks are separated by the configured separator."""
        sep = "\n---\n"
        builder = ContextBuilder(max_tokens=50000, separator=sep)
        result = builder.build(sample_chunks)
        assert sep in result

    def test_contains_score_in_header(self, sample_chunks):
        """Score appears in the chunk header."""
        builder = ContextBuilder(max_tokens=50000)
        result = builder.build(sample_chunks)
        assert "Score:" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_list_returns_empty_string(self):
        """Empty chunk list returns empty string."""
        builder = ContextBuilder()
        assert builder.build([]) == ""

    def test_none_raises_value_error(self):
        """None input raises ValueError."""
        builder = ContextBuilder()
        with pytest.raises(ValueError, match="must not be None"):
            builder.build(None)

    def test_single_chunk(self):
        """A single chunk is formatted correctly."""
        chunk = Document(
            page_content="The only chunk.",
            metadata={"source": "single.pdf", "page": 1, "relevance_score": 0.95},
        )
        builder = ContextBuilder(max_tokens=5000)
        result = builder.build([chunk])
        assert "Chunk 1" in result
        assert "The only chunk." in result
        assert "single.pdf" in result
