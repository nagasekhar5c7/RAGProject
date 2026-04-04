"""
test_query_understanding.py — Tests for QueryUnderstanding (clean + rewrite).

Uses FakeLLMService — no API keys or network needed.
"""

import pytest
from query_understanding import QueryUnderstanding


class TestClean:
    """Tests for QueryUnderstanding.clean() — rule-based cleaning."""

    def test_collapses_whitespace(self, fake_llm):
        """Multiple spaces/tabs between words are collapsed to single space."""
        qu = QueryUnderstanding(llm=fake_llm)
        result = qu.clean("  what   is   attention   mechanism  ")
        assert result == "what is attention mechanism"

    def test_lowercases_query(self, fake_llm):
        """Query is lowercased after cleaning."""
        qu = QueryUnderstanding(llm=fake_llm)
        result = qu.clean("What Is ATTENTION?")
        assert result == "what is attention?"

    def test_strips_leading_trailing_whitespace(self, fake_llm):
        """Leading and trailing whitespace is removed."""
        qu = QueryUnderstanding(llm=fake_llm)
        result = qu.clean("   hello world   ")
        assert result == "hello world"

    def test_preserves_punctuation(self, fake_llm):
        """Question marks, periods, exclamation marks are preserved."""
        qu = QueryUnderstanding(llm=fake_llm)
        assert qu.clean("what is attention?") == "what is attention?"
        assert qu.clean("tell me about transformers.") == "tell me about transformers."
        assert qu.clean("wow!") == "wow!"

    def test_collapses_tabs_and_newlines(self, fake_llm):
        """Tabs and newlines are treated as whitespace and collapsed."""
        qu = QueryUnderstanding(llm=fake_llm)
        result = qu.clean("what\t\tis\nattention")
        assert result == "what is attention"

    def test_empty_query_raises_value_error(self, fake_llm):
        """Empty or whitespace-only query raises ValueError."""
        qu = QueryUnderstanding(llm=fake_llm)
        with pytest.raises(ValueError, match="empty"):
            qu.clean("")

    def test_whitespace_only_raises_value_error(self, fake_llm):
        """All-whitespace query raises ValueError."""
        qu = QueryUnderstanding(llm=fake_llm)
        with pytest.raises(ValueError, match="empty"):
            qu.clean("     ")

    def test_single_word_query(self, fake_llm):
        """Single word query is cleaned and lowercased."""
        qu = QueryUnderstanding(llm=fake_llm)
        assert qu.clean("  TRANSFORMERS  ") == "transformers"


class TestRewrite:
    """Tests for QueryUnderstanding.rewrite() — LLM-based rewriting."""

    def test_rewrite_calls_llm_with_cleaned_query(self, fake_llm):
        """rewrite() sends the cleaned (not raw) query to the LLM."""
        qu = QueryUnderstanding(llm=fake_llm)
        qu.rewrite("  WHAT   IS   ATTENTION?  ")

        # The user message sent to the LLM should be the cleaned version
        user_msg = fake_llm.last_messages[-1]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "what is attention?"

    def test_rewrite_returns_llm_response(self, fake_llm):
        """rewrite() returns whatever the LLM returns."""
        fake_llm.response = "how does the attention mechanism work in transformers?"
        qu = QueryUnderstanding(llm=fake_llm)
        result = qu.rewrite("attention transformers")
        assert result == "how does the attention mechanism work in transformers?"

    def test_rewrite_strips_llm_response(self, fake_llm):
        """rewrite() strips leading/trailing whitespace from LLM output."""
        fake_llm.response = "  cleaned query  \n"
        qu = QueryUnderstanding(llm=fake_llm)
        result = qu.rewrite("some query")
        assert result == "cleaned query"

    def test_rewrite_sends_system_prompt(self, fake_llm):
        """rewrite() includes a system message with the rewrite prompt."""
        qu = QueryUnderstanding(llm=fake_llm)
        qu.rewrite("test query")

        system_msg = fake_llm.last_messages[0]
        assert system_msg["role"] == "system"
        assert "search query reviewer" in system_msg["content"]

    def test_rewrite_empty_query_raises(self, fake_llm):
        """rewrite() raises ValueError for empty query (via clean())."""
        qu = QueryUnderstanding(llm=fake_llm)
        with pytest.raises(ValueError, match="empty"):
            qu.rewrite("   ")
