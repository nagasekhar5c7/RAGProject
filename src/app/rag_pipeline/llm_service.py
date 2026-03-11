"""
llm_service.py — Step 6: LLM Call

Responsibility:
    Send a list of messages to a Groq-hosted open-source LLM via the
    LangChain ChatGroq integration and return the generated text response.

Model:
    llama-3.3-70b-versatile  (default — change in config.py)

    Other available Groq open-source models:
        llama3-8b-8192
        llama3-70b-8192
        mixtral-8x7b-32768
        gemma2-9b-it

Input  : messages (list[dict])  — e.g. [{"role": "user", "content": "..."}]
Output : str                    — the model's response text

API Key:
    Set GROQ_API_KEY in config/config.py
    Get your free key at: https://console.groq.com
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
from config.config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, GROQ_API_KEY

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _to_langchain_messages(messages: list[dict]) -> list:
    """
    Convert OpenAI-style message dicts to LangChain message objects.

    Args:
        messages (list[dict]): Messages with "role" and "content" keys.
            Supported roles: "system", "user", "assistant".

    Returns:
        list: LangChain message objects (SystemMessage, HumanMessage, AIMessage).

    Raises:
        ValueError: If an unsupported role is encountered.
    """
    role_map = {
        "system":    SystemMessage,
        "user":      HumanMessage,
        "assistant": AIMessage,
    }
    converted = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in role_map:
            raise ValueError(f"Unsupported message role: '{role}'. Use system/user/assistant.")
        converted.append(role_map[role](content=content))
    return converted


# ──────────────────────────────────────────────
# LLM Service
# ──────────────────────────────────────────────

class LLMService:
    """
    Wraps LangChain's ChatGroq to call Groq-hosted open-source LLMs via API.

    The model is loaded once at construction time. All parameters are
    read from config.py and overridable via constructor arguments —
    swapping models is a config change, not a code change.

    Attributes:
        model (str):         Groq model identifier.
        temperature (float): Sampling temperature. 0.0 = deterministic.
        max_tokens (int):    Maximum tokens in the generated response.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ):
        """
        Initialize the ChatGroq client.

        Args:
            model (str):         Groq model name. Defaults to config LLM_MODEL.
            temperature (float): Sampling temperature. Defaults to 0.0.
            max_tokens (int):    Max response tokens. Defaults to 1024.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = ChatGroq(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=GROQ_API_KEY,
        )

        logger.info(
            f"LLMService initialized — model='{self.model}', "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    def generate(self, messages: list[dict]) -> str:
        """
        Send messages to the Groq LLM and return the response text.

        Args:
            messages (list[dict]): Conversation messages in OpenAI-style format:
                [
                    {"role": "system",  "content": "You are a helpful assistant."},
                    {"role": "user",    "content": "What is attention?"},
                ]

        Returns:
            str: The model's generated response text.

        Raises:
            ValueError:   If the messages list is empty or contains an invalid role.
            RuntimeError: If the Groq API call fails (network error, invalid key, etc.).
        """
        if not messages:
            raise ValueError("messages list is empty — nothing to send to the LLM.")

        logger.info(f"Calling Groq model '{self.model}' with {len(messages)} message(s)...")

        try:
            lc_messages = _to_langchain_messages(messages)
            response = self._client.invoke(lc_messages)
            reply = response.content
            logger.info(f"Response received — {len(reply)} character(s).")
            return reply

        except Exception as e:
            raise RuntimeError(
                f"Groq API call failed for model '{self.model}': {e}"
            ) from e


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test — sends a single user message to the Groq LLM.

    Prerequisites:
        export GROQ_API_KEY="your_key_here"

    Run from the project root:
        uv run python src/app/rag_pipeline/llm_service.py
    """
    llm = LLMService()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer concisely.",
        },
        {
            "role": "user",
            "content": "What is a transformer model in machine learning? Answer in 2 sentences.",
        },
    ]

    print(f"\nModel  : {llm.model}")
    print(f"Prompt : {messages[-1]['content']}")
    print("-" * 50)

    response = llm.generate(messages)
    print(f"Response:\n{response}")
