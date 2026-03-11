"""
prompt_builder.py — Step 5: Prompt Assembly & LLM Call

Responsibility:
    Accept the rewritten query (from query_understanding.py) and the
    formatted context block (from context_builder.py), assemble them into
    an LLM message list, call LLMService, and return the generated answer.

Two components:
    SYSTEM_PROMPT_TEMPLATE  — jinja-style template; {context} is replaced at
                              call time with the formatted context block.
    PromptBuilder           — builds the messages list and calls LLMService.

Input  : query   (str) — rewritten query from query_understanding.py
         context (str) — formatted context block from context_builder.py
Output : str           — the LLM-generated answer

Message structure sent to the LLM:
    [
        {"role": "system",  "content": <system prompt with context injected>},
        {"role": "user",    "content": <query>},
    ]

System prompt behaviour:
    - Answer ONLY from the provided context chunks.
    - If the answer is not in the context, say so explicitly.
    - Cite the source file and page number where relevant.
    - Be concise and factual; avoid speculation.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

from llm_service import LLMService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# System Prompt Template
# ──────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a precise, factual question-answering assistant for a book knowledge base.

You are given a set of relevant excerpts (context) retrieved from the knowledge base.
Your job is to answer the user's question using ONLY the information in the context below.

Rules:
- Answer strictly from the provided context. Do NOT use outside knowledge.
- If the answer is fully or partially present in the context, provide it clearly and concisely.
- If the context does not contain enough information to answer, respond with:
  "I could not find a sufficient answer in the provided context."
- Where possible, cite the source (e.g. "According to [Source: book.pdf | Page: 42]...").
- Do NOT speculate, hallucinate, or add information not present in the context.
- Keep your answer focused and factual.

--- CONTEXT START ---
{context}
--- CONTEXT END ---
"""


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

class PromptBuilder:
    """
    Assembles the LLM message list from a query and context block,
    then calls LLMService to generate the final answer.

    The system message contains the full context block injected into
    SYSTEM_PROMPT_TEMPLATE.  The user message contains the raw query.

    LLMService is injected at construction time so it can be swapped
    with a mock during testing without changing this class.

    Attributes:
        llm (LLMService): The LLM client used to generate the answer.
    """

    def __init__(self, llm: LLMService):
        """
        Args:
            llm (LLMService): Initialised LLM service used to generate answers.
        """
        self.llm = llm
        logger.info("PromptBuilder initialised.")

    def build_messages(self, query: str, context: str) -> list[dict]:
        """
        Assemble the messages list to send to the LLM.

        The system message has the context injected into
        SYSTEM_PROMPT_TEMPLATE via a simple str.format() substitution.
        The user message is the raw query string.

        Args:
            query   (str): The rewritten query from query_understanding.py.
            context (str): The formatted context block from context_builder.py.

        Returns:
            list[dict]: OpenAI-style message list:
                [
                    {"role": "system", "content": <system prompt + context>},
                    {"role": "user",   "content": <query>},
                ]

        Raises:
            ValueError: If query or context is empty.
        """
        if not query.strip():
            raise ValueError("Query is empty — cannot build prompt.")
        if not context.strip():
            raise ValueError(
                "Context is empty — no content to answer from. "
                "Ensure the retriever and context builder returned results."
            )

        system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": query},
        ]

        logger.info(
            f"Prompt built — system ~{len(system_content) // 4} tokens, "
            f"query '{query[:60]}{'...' if len(query) > 60 else ''}'"
        )

        return messages

    def generate(self, query: str, context: str) -> str:
        """
        Build the prompt and call the LLM to generate an answer.

        Convenience method that calls build_messages() and then passes
        the result to LLMService.generate().

        Args:
            query   (str): The rewritten query from query_understanding.py.
            context (str): The formatted context block from context_builder.py.

        Returns:
            str: The LLM-generated answer text.

        Raises:
            ValueError:   If query or context is empty.
            RuntimeError: If the LLM call fails.
        """
        messages = self.build_messages(query, context)

        logger.info("Calling LLM to generate answer...")
        answer = self.llm.generate(messages)
        logger.info(f"Answer received — {len(answer)} character(s).")

        return answer


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test — runs the full pipeline through prompt generation.
    Requires a built FAISS index and valid API keys in config/config.py.

    Run from the project root:
        uv run python src/app/rag_pipeline/prompt_builder.py
    """
    from query_understanding import QueryUnderstanding
    from retriever import Retriever
    from reranker import Reranker
    from context_builder import ContextBuilder

    raw_query = "what is the attention mechanism in transformers?"

    # Step 1 — clean & rewrite
    llm = LLMService()
    qu = QueryUnderstanding(llm=llm)
    rewritten_query = qu.rewrite(raw_query)

    # Step 2 — retrieve
    retriever = Retriever()
    chunks = retriever.retrieve(rewritten_query)

    # Step 3 — rerank
    reranker = Reranker()
    top_chunks = reranker.rerank(rewritten_query, chunks)

    # Step 4 — build context
    context = ContextBuilder().build(top_chunks)

    # Step 5 — build prompt & generate answer
    prompt_builder = PromptBuilder(llm=llm)
    answer = prompt_builder.generate(query=rewritten_query, context=context)

    print(f"\n{'='*60}")
    print(f"Raw query      : {raw_query}")
    print(f"Rewritten query: {rewritten_query}")
    print(f"Chunks used    : {len(top_chunks)}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{answer}")
    print(f"{'='*60}")
