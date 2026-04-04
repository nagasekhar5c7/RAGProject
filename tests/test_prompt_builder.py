"""
test_prompt_builder.py — Tests for PromptBuilder.

Tests message assembly and LLM call delegation.
Uses FakeLLMService — no API keys needed.
"""

import pytest
from prompt_builder import PromptBuilder, SYSTEM_PROMPT_TEMPLATE


class TestBuildMessages:
    """Tests for PromptBuilder.build_messages()."""

    def test_returns_two_messages(self, fake_llm):
        """build_messages() returns exactly [system, user] messages."""
        pb = PromptBuilder(llm=fake_llm)
        messages = pb.build_messages("test query?", "some context here")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_context_injected_into_system_prompt(self, fake_llm):
        """The context string appears inside the system message."""
        pb = PromptBuilder(llm=fake_llm)
        context = "The attention mechanism focuses on relevant tokens."
        messages = pb.build_messages("what is attention?", context)
        assert context in messages[0]["content"]

    def test_query_is_user_message(self, fake_llm):
        """The user query is the content of the user message."""
        pb = PromptBuilder(llm=fake_llm)
        messages = pb.build_messages("what is attention?", "some context")
        assert messages[1]["content"] == "what is attention?"

    def test_system_prompt_contains_rules(self, fake_llm):
        """The system message contains the grounding rules."""
        pb = PromptBuilder(llm=fake_llm)
        messages = pb.build_messages("query", "context")
        system_content = messages[0]["content"]
        assert "ONLY" in system_content
        assert "CONTEXT START" in system_content
        assert "CONTEXT END" in system_content

    def test_empty_query_raises(self, fake_llm):
        """Empty query raises ValueError."""
        pb = PromptBuilder(llm=fake_llm)
        with pytest.raises(ValueError, match="Query is empty"):
            pb.build_messages("", "some context")

    def test_empty_context_raises(self, fake_llm):
        """Empty context raises ValueError."""
        pb = PromptBuilder(llm=fake_llm)
        with pytest.raises(ValueError, match="Context is empty"):
            pb.build_messages("what is attention?", "")

    def test_whitespace_only_query_raises(self, fake_llm):
        """Whitespace-only query raises ValueError."""
        pb = PromptBuilder(llm=fake_llm)
        with pytest.raises(ValueError, match="Query is empty"):
            pb.build_messages("   ", "some context")


class TestGenerate:
    """Tests for PromptBuilder.generate()."""

    def test_returns_llm_response(self, fake_llm):
        """generate() returns the LLM's answer."""
        fake_llm.response = "The attention mechanism allows models to focus on relevant parts."
        pb = PromptBuilder(llm=fake_llm)
        answer = pb.generate("what is attention?", "context about attention")
        assert answer == "The attention mechanism allows models to focus on relevant parts."

    def test_passes_messages_to_llm(self, fake_llm):
        """generate() sends the correct messages to the LLM."""
        pb = PromptBuilder(llm=fake_llm)
        pb.generate("my question?", "my context")

        assert fake_llm.last_messages is not None
        assert len(fake_llm.last_messages) == 2
        assert fake_llm.last_messages[1]["content"] == "my question?"

    def test_empty_query_raises(self, fake_llm):
        """generate() raises ValueError for empty query."""
        pb = PromptBuilder(llm=fake_llm)
        with pytest.raises(ValueError):
            pb.generate("", "some context")

    def test_empty_context_raises(self, fake_llm):
        """generate() raises ValueError for empty context."""
        pb = PromptBuilder(llm=fake_llm)
        with pytest.raises(ValueError):
            pb.generate("some query?", "  ")
