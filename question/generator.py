"""
question/generator.py
KIMI LLM integration — sync + streaming (SSE).
Delegates to shared.llm for client management and retry logic.
"""

import logging
from typing import Generator

from shared.llm import chat_stream
from config import KIMI_MODEL_QA, LLM_TEMPERATURE, LLM_MAX_TOKENS
from question.prompts import SYSTEM_PROMPT, build_qa_prompt

logger = logging.getLogger(__name__)


def generate_answer_stream(
    query: str,
    context: str,
    data_freshness: str = "",
) -> Generator[tuple, None, None]:
    """Generate answer in streaming mode — yields (type, text) tuples for SSE."""
    user_prompt = build_qa_prompt(query, context, data_freshness)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    yield from chat_stream(
        messages=messages,
        model=KIMI_MODEL_QA,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
