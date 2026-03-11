"""
context_builder.py — Step 4: Context Assembly

Responsibility:
    Accept the reranked list of Documents from reranker.py, clean them,
    remove duplicates, enforce a token budget, and format them into a
    single context string ready to be injected into the LLM prompt.

Pipeline (in order):
    1. clean        — normalize whitespace, strip artifacts from each chunk
    2. deduplicate  — fingerprint each chunk (SHA-256 of normalised text),
                      drop any chunk whose fingerprint has already been seen
    3. order        — sort surviving chunks by relevance_score (highest first)
    4. trim         — accumulate chunks until the token budget is exhausted;
                      drop any chunk that would exceed the budget
    5. format       — render each chunk with a metadata header and join with
                      a configurable separator

Input  : chunks (list[Document]) — reranked chunks from reranker.py
Output : str                     — formatted context block for the LLM prompt

Token estimation:
    No external tokeniser is required.  Tokens are estimated as:
        estimated_tokens = len(text) // 4
    (approximately 4 characters per token for English prose)

Config:
    CONTEXT_MAX_TOKENS      = 2000              (token budget for the whole block)
    CONTEXT_CHUNK_SEPARATOR = "\\n\\n---\\n\\n" (separator between chunk blocks)
"""

import hashlib
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
from config.config import (
    CONTEXT_CHUNK_SEPARATOR,
    CONTEXT_MAX_TOKENS,
)

from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Context Builder
# ──────────────────────────────────────────────

class ContextBuilder:
    """
    Assembles a clean, deduplicated, token-budgeted context string from a
    list of reranked Document chunks.

    Each chunk is processed through five sequential stages:
        clean → deduplicate → order → trim → format

    The resulting string is suitable for direct injection into an LLM prompt
    as the ``{context}`` placeholder.

    Attributes:
        max_tokens (int):  Maximum estimated tokens allowed in the context block.
        separator  (str):  String placed between consecutive formatted chunks.
    """

    def __init__(
        self,
        max_tokens: int = CONTEXT_MAX_TOKENS,
        separator: str = CONTEXT_CHUNK_SEPARATOR,
    ):
        """
        Args:
            max_tokens (int): Token budget for the assembled context.
                              Defaults to CONTEXT_MAX_TOKENS.
            separator  (str): Separator inserted between chunk blocks.
                              Defaults to CONTEXT_CHUNK_SEPARATOR.
        """
        self.max_tokens = max_tokens
        self.separator = separator
        logger.info(
            f"ContextBuilder initialised — max_tokens={max_tokens}"
        )

    # ── Internal helpers ──────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Uses the rule-of-thumb: 1 token ≈ 4 characters for English prose.
        No external tokeniser required.

        Args:
            text (str): The text to estimate.

        Returns:
            int: Estimated token count (>= 0).
        """
        return max(0, len(text) // 4)

    def _clean(self, text: str) -> str:
        """
        Normalize a chunk's raw text.

        Steps applied in order:
            1. Strip leading / trailing whitespace.
            2. Collapse runs of spaces / tabs into a single space.
            3. Collapse runs of 3+ newlines into exactly two newlines
               (preserving intentional paragraph breaks).
            4. Strip non-printable / null characters.

        Args:
            text (str): Raw page_content of a Document chunk.

        Returns:
            str: Cleaned text string.
        """
        # Strip edges
        text = text.strip()
        # Collapse horizontal whitespace runs
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove null / non-printable characters (keep newlines and tabs)
        text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text)
        return text

    def _fingerprint(self, text: str) -> str:
        """
        Compute a deduplication fingerprint for a text string.

        The fingerprint is the SHA-256 hex digest of the lowercase,
        whitespace-collapsed version of the text so that trivial formatting
        differences between copies of the same content are ignored.

        Args:
            text (str): Cleaned chunk text.

        Returns:
            str: 64-character hex digest.
        """
        normalised = re.sub(r"\s+", " ", text.lower()).strip()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def _format_chunk(self, doc: Document, index: int) -> str:
        """
        Render a single Document chunk as a labelled text block.

        The header line contains the most useful metadata fields so the LLM
        can cite sources if needed.  Only non-null fields are shown.

        Format:
            [Chunk 1 | Source: book.pdf | Page: 42 | Score: 0.9200]
            <cleaned page_content>

        Args:
            doc   (Document): The chunk to format.
            index (int):      1-based position in the final context (for readability).

        Returns:
            str: Formatted chunk string.
        """
        meta = doc.metadata

        parts = [f"Chunk {index}"]

        source = meta.get("source")
        if source:
            parts.append(f"Source: {Path(source).name}")

        page = meta.get("page")
        if page is not None:
            parts.append(f"Page: {page}")

        row = meta.get("row")
        if row is not None:
            parts.append(f"Row: {row}")

        score = meta.get("relevance_score") or meta.get("similarity_score")
        if score is not None:
            parts.append(f"Score: {score:.4f}")

        header = "[" + " | ".join(parts) + "]"
        return f"{header}\n{doc.page_content}"

    # ── Public API ────────────────────────────────────────────

    def build(self, chunks: list[Document]) -> str:
        """
        Assemble a context string from a list of reranked Documents.

        Pipeline stages:
            1. **Clean**       — normalise whitespace / strip artifacts.
            2. **Deduplicate** — drop chunks whose normalised content has
                                 already been seen (SHA-256 fingerprint).
            3. **Order**       — sort by relevance_score desc (ties broken by
                                 similarity_score desc).
            4. **Trim**        — accumulate chunks until the token budget
                                 (max_tokens) is reached; drop the rest.
            5. **Format**      — render each chunk with a metadata header and
                                 join with the configured separator.

        Args:
            chunks (list[Document]): Reranked chunks from reranker.py.

        Returns:
            str: Formatted context block.  Returns an empty string if no
                 chunks survive deduplication / trimming.

        Raises:
            ValueError: If chunks is None.
        """
        if chunks is None:
            raise ValueError("chunks must not be None.")

        if not chunks:
            logger.warning("ContextBuilder received an empty chunk list. Returning empty context.")
            return ""

        logger.info(f"Building context from {len(chunks)} chunk(s)...")

        # ── Stage 1: Clean ────────────────────────────────────
        for doc in chunks:
            doc.page_content = self._clean(doc.page_content)

        after_clean = len(chunks)
        logger.info(f"Stage 1 — Clean: {after_clean} chunk(s) remain.")

        # ── Stage 2: Deduplicate ──────────────────────────────
        seen_fingerprints: set[str] = set()
        unique_chunks: list[Document] = []

        for doc in chunks:
            fp = self._fingerprint(doc.page_content)
            if fp in seen_fingerprints:
                logger.debug(
                    f"Duplicate dropped — source={doc.metadata.get('source', '?')}, "
                    f"fingerprint={fp[:12]}..."
                )
            else:
                seen_fingerprints.add(fp)
                unique_chunks.append(doc)

        dropped = after_clean - len(unique_chunks)
        logger.info(
            f"Stage 2 — Deduplicate: {dropped} duplicate(s) dropped, "
            f"{len(unique_chunks)} unique chunk(s) remain."
        )

        # ── Stage 3: Order ────────────────────────────────────
        unique_chunks.sort(
            key=lambda d: (
                d.metadata.get("relevance_score", 0.0),
                d.metadata.get("similarity_score", 0.0),
            ),
            reverse=True,
        )
        logger.info("Stage 3 — Order: chunks sorted by relevance_score descending.")

        # ── Stage 4: Trim to token budget ─────────────────────
        trimmed_chunks: list[Document] = []
        running_tokens = 0

        for doc in unique_chunks:
            chunk_tokens = self._estimate_tokens(doc.page_content)
            if running_tokens + chunk_tokens > self.max_tokens:
                logger.debug(
                    f"Token budget reached at {running_tokens}/{self.max_tokens}. "
                    f"Dropping remaining {len(unique_chunks) - len(trimmed_chunks)} chunk(s)."
                )
                break
            trimmed_chunks.append(doc)
            running_tokens += chunk_tokens

        logger.info(
            f"Stage 4 — Trim: {len(trimmed_chunks)} chunk(s) kept "
            f"(~{running_tokens} tokens of {self.max_tokens} budget used)."
        )

        if not trimmed_chunks:
            logger.warning("All chunks were trimmed by token budget. Returning empty context.")
            return ""

        # ── Stage 5: Format ───────────────────────────────────
        formatted_blocks = [
            self._format_chunk(doc, i + 1)
            for i, doc in enumerate(trimmed_chunks)
        ]
        context = self.separator.join(formatted_blocks)

        logger.info(
            f"Stage 5 — Format: context assembled "
            f"({len(trimmed_chunks)} chunk(s), ~{self._estimate_tokens(context)} tokens)."
        )

        return context


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test.  Requires a built FAISS index and valid API keys.

    Run from the project root:
        uv run python src/app/rag_pipeline/context_builder.py
    """
    from llm_service import LLMService
    from query_understanding import QueryUnderstanding
    from retriever import Retriever
    from reranker import Reranker

    raw_query = "what is the attention mechanism in transformers?"

    # Step 1 — clean & rewrite
    qu = QueryUnderstanding(llm=LLMService())
    rewritten_query = qu.rewrite(raw_query)

    # Step 2 — retrieve
    retriever = Retriever()
    chunks = retriever.retrieve(rewritten_query)

    # Step 3 — rerank
    reranker = Reranker()
    top_chunks = reranker.rerank(rewritten_query, chunks)

    # Step 4 — build context
    builder = ContextBuilder()
    context = builder.build(top_chunks)

    print(f"\n{'='*60}")
    print(f"Query          : {raw_query}")
    print(f"Chunks in      : {len(top_chunks)}")
    print(f"Context tokens : ~{len(context) // 4}")
    print(f"{'='*60}")
    print("\n--- CONTEXT BLOCK ---\n")
    print(context)
    print("\n--- END CONTEXT ---")
