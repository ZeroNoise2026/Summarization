"""
summarizer.py
Moonshot (Kimi) API integration for generating analysis reports.
Uses the OpenAI-compatible API format.
"""

import logging
import time
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

from config import MOONSHOT_API_KEY, MOONSHOT_BASE_URL, MOONSHOT_MODEL
from prompts import SYSTEM_PROMPT, build_user_prompt
from fetcher import TickerContext

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not MOONSHOT_API_KEY:
            raise ValueError("MOONSHOT_API_KEY is not set. Check your .env file.")
        _client = OpenAI(
            api_key=MOONSHOT_API_KEY,
            base_url=MOONSHOT_BASE_URL,
        )
    return _client


def generate_summary(ctx: TickerContext, max_retries: int = 3) -> str:
    """Send context to Moonshot API and return the generated analysis report."""
    client = _get_client()
    user_prompt = build_user_prompt(ctx)

    logger.info(f"Calling Moonshot API for {ctx.ticker} (model={MOONSHOT_MODEL}, prompt ~{len(user_prompt):,} chars)")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MOONSHOT_MODEL,
                messages=messages,
                temperature=1.0,
            )
            content = response.choices[0].message.content
            usage = response.usage
            logger.info(
                f"  {ctx.ticker} done: "
                f"prompt_tokens={usage.prompt_tokens}, "
                f"completion_tokens={usage.completion_tokens}, "
                f"total_tokens={usage.total_tokens}"
            )
            return content

        except RateLimitError as e:
            wait = min(2 ** attempt * 10, 60)
            logger.warning(f"  Rate limited (attempt {attempt}/{max_retries}), waiting {wait}s: {e}")
            time.sleep(wait)

        except (APITimeoutError, APIConnectionError) as e:
            wait = 2 ** attempt * 5
            logger.warning(f"  Connection issue (attempt {attempt}/{max_retries}), retrying in {wait}s: {e}")
            time.sleep(wait)

        except Exception as e:
            logger.error(f"  Unexpected error for {ctx.ticker}: {e}")
            raise

    raise RuntimeError(f"Failed to generate summary for {ctx.ticker} after {max_retries} attempts")
