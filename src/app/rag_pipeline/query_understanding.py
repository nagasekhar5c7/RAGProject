"""
query_understanding.py — Step 1: Query Cleaning & Rewriting

Responsibility:
    Accept a raw user query, clean it, and rewrite it for better retrieval.

Two functions:
    1. clean(query)   — strips extra whitespace, normalizes spaces, lowercases
    2. rewrite(query) — uses the LLM (via LLMService) to expand/rephrase the
                        query into a retrieval-optimized version

Input  : raw query string
Output : cleaned query string  (from clean)
         rewritten query string (from rewrite — LLM output)

Design:
    QueryUnderstanding
    ├── clean(query: str)   → str
    └── rewrite(query: str) → str   (calls LLMService internally)
"""

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

from llm_service import LLMService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# System prompt used for LLM-based query rewriting
REWRITE_SYSTEM_PROMPT = (
    "You are a search query reviewer for a retrieval-augmented generation system. "
    "Your job is to evaluate whether the user's query needs improvement. "
    "Rules:\n"
    "- If the query is already clear, specific, and self-contained → return it EXACTLY as given, word for word.\n"
    "- Only rewrite if the query is vague, has typos, is too short to be meaningful, or is ambiguous.\n"
    "- When you do rewrite, make minimal targeted changes — do NOT expand, add keywords, or make it longer.\n"
    "- ALWAYS preserve the original punctuation at the end of the query (e.g. ?, !, .).\n"
    "- Do NOT add, remove, or change any punctuation unless fixing a typo.\n"
    "- Return ONLY the final query string — no explanation, no preamble, no quotes, no bullet points."
)


# ──────────────────────────────────────────────
# Query Understanding
# ──────────────────────────────────────────────

class QueryUnderstanding:
    """
    Cleans and rewrites a raw user query before it is sent to the retriever.

    clean()   — rule-based: strips whitespace, collapses spaces, lowercases.
    rewrite() — LLM-based: sends the cleaned query to the LLM with a
                rewrite prompt and returns the optimized query string.

    The LLMService is injected at construction time so it can be swapped
    with a mock in tests without changing this class.

    Attributes:
        llm (LLMService): The LLM client used for query rewriting.
    """

    def __init__(self, llm: LLMService):
        """
        Args:
            llm (LLMService): Initialized LLM service used for rewriting.
        """
        self.llm = llm
        logger.info("QueryUnderstanding initialized.")

    def clean(self, query: str) -> str:
        """
        Clean the raw query with rule-based text normalization.

        Steps applied in order:
            1. Strip leading and trailing whitespace.
            2. Collapse all internal whitespace runs (spaces, tabs, newlines)
               into a single space.
            3. Lowercase the entire string.

        Args:
            query (str): The raw user query.

        Returns:
            str: The cleaned, normalized query.

        Raises:
            ValueError: If the query is empty or blank after cleaning.
        """
        logger.info(f"Cleaning query: '{query[:80]}{'...' if len(query) > 80 else ''}'")

        # Strip + collapse whitespace + lowercase
        cleaned = re.sub(r"\s+", " ", query.strip()).lower()

        if not cleaned:
            raise ValueError("Query is empty after cleaning.")

        logger.info(f"Cleaned query : '{cleaned}'")
        return cleaned

    def rewrite(self, query: str) -> str:
        """
        Rewrite the query using the LLM to improve retrieval quality.

        First cleans the query, then sends it to the LLM with a system
        prompt that instructs it to return only the rewritten query text.

        Args:
            query (str): The raw or pre-cleaned user query.

        Returns:
            str: The LLM-rewritten query, stripped of any surrounding whitespace.

        Raises:
            ValueError:   If the query is empty or blank after cleaning.
            RuntimeError: If the LLM call fails.
        """
        cleaned = self.clean(query)

        logger.info(f"Rewriting query via LLM...")

        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user",   "content": cleaned},
        ]

        rewritten = self.llm.generate(messages).strip()

        logger.info(f"Rewritten query: '{rewritten}'")
        return rewritten


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test for both clean() and rewrite().

    Run from the project root:
        uv run python src/app/rag_pipeline/query_understanding.py
    """
    raw_query = "  what   is   attention   mechanism    in   transformers?  "

    # ── Test clean ────────────────────────────────────────────
    qu = QueryUnderstanding(llm=LLMService())

    cleaned = qu.clean(raw_query)

    print(f"\n{'='*50}")
    print(f"Raw query     : '{raw_query}'")
    print(f"Cleaned query : '{cleaned}'")

    # ── Test rewrite ──────────────────────────────────────────
    rewritten = qu.rewrite(raw_query)

    print(f"Rewritten     : '{rewritten}'")
    print(f"{'='*50}")
