"""
summary/summarizer.py
Moonshot (Kimi) API integration for generating analysis reports.
Delegates to shared.llm for client management and retry logic.
"""

import logging

from config import MOONSHOT_MODEL
from shared.llm import chat
from summary.prompts import SYSTEM_PROMPT, build_user_prompt
from summary.fetcher import TickerContext

logger = logging.getLogger(__name__)


def generate_summary(ctx: TickerContext, max_retries: int = 3) -> str:
    """Send context to Moonshot API and return the generated analysis report."""
    user_prompt = build_user_prompt(ctx)

    logger.info(f"Calling Moonshot API for {ctx.ticker} (model={MOONSHOT_MODEL}, prompt ~{len(user_prompt):,} chars)")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    content, usage = chat(
        messages=messages,
        model=MOONSHOT_MODEL,
        temperature=1.0,
        max_retries=max_retries,
    )
    logger.info(
        f"  {ctx.ticker} done: "
        f"prompt_tokens={usage['prompt_tokens']}, "
        f"completion_tokens={usage['completion_tokens']}, "
        f"total_tokens={usage['total_tokens']}"
    )
    return content
