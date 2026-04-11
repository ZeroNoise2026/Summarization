"""
summary/summarizer.py
Moonshot (Kimi) API integration for generating analysis reports.
Delegates to shared.llm for client management and retry logic.

Model selection: automatically picks the smallest (fastest) model
whose context window can fit the prompt, then falls back to the
largest model for very long contexts.
"""

import logging

from shared.llm import chat, pick_model
from summary.prompts import SYSTEM_PROMPT, build_user_prompt
from summary.fetcher import TickerContext

logger = logging.getLogger(__name__)


def generate_summary(ctx: TickerContext, max_retries: int = 3) -> str:
    """Send context to Moonshot API and return the generated analysis report."""
    user_prompt = build_user_prompt(ctx)
    total_chars = len(SYSTEM_PROMPT) + len(user_prompt)
    model = pick_model(total_chars)

    logger.info(
        f"Calling Moonshot API for {ctx.ticker} "
        f"(model={model}, prompt ~{total_chars:,} chars)"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    content, usage = chat(
        messages=messages,
        model=model,
        temperature=1.0,
        max_retries=max_retries,
    )
    logger.info(
        f"  {ctx.ticker} done: "
        f"model={model}, "
        f"prompt_tokens={usage['prompt_tokens']}, "
        f"completion_tokens={usage['completion_tokens']}, "
        f"total_tokens={usage['total_tokens']}"
    )
    return content
