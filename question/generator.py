"""
question/generator.py
KIMI LLM integration — sync + streaming (SSE).
Delegates to shared.llm for client management and retry logic.
"""

import logging
from typing import Generator

from shared.llm import chat_stream, pick_model
from config import LLM_TEMPERATURE, LLM_MAX_TOKENS
from question.prompts import SYSTEM_PROMPT, build_qa_prompt

logger = logging.getLogger(__name__)


def generate_answer_stream(
    query: str,
    context: str,
    data_freshness: str = "",
) -> Generator[tuple, None, None]:
    """Generate answer in streaming mode — yields (type, text) tuples for SSE."""
    user_prompt = build_qa_prompt(query, context, data_freshness)
    total_chars = len(SYSTEM_PROMPT) + len(user_prompt)
    model = pick_model(total_chars)

    logger.info(f"QA stream: model={model}, prompt ~{total_chars:,} chars")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    yield from chat_stream(
        messages=messages,
        model=model,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
