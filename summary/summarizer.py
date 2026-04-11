"""
summary/summarizer.py
Moonshot (Kimi) API integration for generating analysis reports.
Delegates to shared.llm for client management and retry logic.

Always uses kimi-k2.5 (reasoning model) for investment analysis —
quality matters more than speed here.
"""

import logging

from shared.llm import chat
from summary.prompts import SYSTEM_PROMPT, build_user_prompt
from summary.fetcher import TickerContext

logger = logging.getLogger(__name__)

# Summarization always uses the reasoning model for best analysis quality.
_SUMMARY_MODEL = "kimi-k2.5"


def generate_summary(ctx: TickerContext, max_retries: int = 3) -> str:
    """Send context to Moonshot API and return the generated analysis report."""
    user_prompt = build_user_prompt(ctx)
    total_chars = len(SYSTEM_PROMPT) + len(user_prompt)

    logger.info(
        f"Calling Moonshot API for {ctx.ticker} "
        f"(model={_SUMMARY_MODEL}, prompt ~{total_chars:,} chars)"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    content, usage = chat(
        messages=messages,
        model=_SUMMARY_MODEL,
        temperature=1.0,
        max_retries=max_retries,
    )
    logger.info(
        f"  {ctx.ticker} done: "
        f"model={_SUMMARY_MODEL}, "
        f"prompt_tokens={usage['prompt_tokens']}, "
        f"completion_tokens={usage['completion_tokens']}, "
        f"total_tokens={usage['total_tokens']}"
    )
    return content
