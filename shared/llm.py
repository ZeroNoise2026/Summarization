"""
shared/llm.py
Shared KIMI (Moonshot) LLM client — singleton + retry logic.

Used by:
  - summary/summarizer.py  (report generation, kimi-k2.5, temp=1.0)
  - question/generator.py  (QA generation, kimi-k2.5, temp=1.0)
  - question/router.py     (L2 classification, moonshot-v1-8k, temp=0, max_tokens=150)

Note: kimi-k2.5 is a reasoning model — only accepts temperature=1.
"""

import logging
import time
from typing import Generator, Optional

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, AuthenticationError

from config import MOONSHOT_API_KEY, MOONSHOT_BASE_URL

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Lazy singleton — one OpenAI client shared across the entire process."""
    global _client
    if _client is None:
        if not MOONSHOT_API_KEY:
            raise ValueError("MOONSHOT_API_KEY is not set. Check your .env file.")
        _client = OpenAI(
            api_key=MOONSHOT_API_KEY,
            base_url=MOONSHOT_BASE_URL,
        )
    return _client


def has_api_key() -> bool:
    """Check if MOONSHOT_API_KEY is configured (without raising)."""
    return bool(MOONSHOT_API_KEY)


def chat(
    messages: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    timeout: Optional[float] = None,
) -> tuple[str, dict]:
    """Send a chat completion request with retry logic.

    Args:
        messages: OpenAI-format messages [{"role": ..., "content": ...}]
        model: Model name (e.g. "moonshot-v1-8k", "moonshot-v1-128k")
        temperature: Sampling temperature
        max_tokens: Max tokens in response (None = model default)
        max_retries: Number of retry attempts for rate-limit / connection errors
        timeout: Request timeout in seconds (None = client default)

    Returns:
        (content, usage_dict) where usage_dict has prompt_tokens, completion_tokens, total_tokens
    """
    client = get_client()

    # kimi-k2.5 only accepts temperature=1
    if "kimi-k2" in model:
        temperature = 1.0

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            response = client.chat.completions.create(**kwargs)
            latency = int((time.time() - t0) * 1000)
            content = response.choices[0].message.content
            usage = response.usage
            usage_dict = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            logger.info(
                f"LLM call done: model={model}, "
                f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, "
                f"total={usage.total_tokens}"
            )
            # Audit
            from audit import log_api_call
            log_api_call(
                service="summary" if "128k" in model else "question",
                api_type="kimi_llm",
                endpoint="chat.completions",
                model=model,
                tokens_in=usage.prompt_tokens,
                tokens_out=usage.completion_tokens,
                tokens_total=usage.total_tokens,
                latency_ms=latency,
            )
            return content, usage_dict

        except RateLimitError as e:
            latency = int((time.time() - t0) * 1000)
            wait = min(2 ** attempt * 10, 60)
            logger.warning(f"Rate limited (attempt {attempt}/{max_retries}), waiting {wait}s: {e}")
            from audit import log_api_call
            log_api_call(
                service="summary" if "128k" in model else "question",
                api_type="kimi_llm", endpoint="chat.completions", model=model,
                latency_ms=latency, status="rate_limited", error_msg=str(e)[:200],
            )
            time.sleep(wait)

        except (APITimeoutError, APIConnectionError) as e:
            wait = 2 ** attempt * 5
            logger.warning(f"Connection issue (attempt {attempt}/{max_retries}), retrying in {wait}s: {e}")
            time.sleep(wait)

        except AuthenticationError as e:
            # Invalid/placeholder API key — don't retry, just raise
            logger.warning(f"LLM authentication failed (invalid API key): {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            raise

    raise RuntimeError(f"LLM call failed after {max_retries} attempts (model={model})")


def chat_stream(
    messages: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
) -> Generator[tuple, None, None]:
    """Send a streaming chat completion request — yields (type, text) tuples.

    Tuple types:
      ("thinking", text) — reasoning tokens (kimi-k2.5 reasoning_content)
      ("token", text)    — final answer content tokens

    Retries on RateLimitError (429) up to max_retries times.
    """
    client = get_client()

    # kimi-k2.5 only accepts temperature=1
    if "kimi-k2" in model:
        temperature = 1.0

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)

            t0 = time.time()
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    # kimi-k2.5: reasoning_content chunks arrive first, then content
                    reasoning = getattr(delta, 'reasoning_content', None) or ''
                    content = delta.content or ''
                    if reasoning:
                        yield ("thinking", reasoning)
                    if content:
                        yield ("token", content)

            latency = int((time.time() - t0) * 1000)
            from audit import log_api_call
            log_api_call(
                service="question",
                api_type="kimi_llm",
                endpoint="chat.completions.stream",
                model=model,
                latency_ms=latency,
            )
            return  # success — exit retry loop

        except RateLimitError as e:
            wait = min(2 ** attempt * 10, 60)
            logger.warning(f"Stream rate limited (attempt {attempt}/{max_retries}), waiting {wait}s: {e}")
            from audit import log_api_call
            log_api_call(
                service="question",
                api_type="kimi_llm", endpoint="chat.completions.stream", model=model,
                status="rate_limited", error_msg=str(e)[:200],
            )
            if attempt == max_retries:
                # Yield error token instead of raising — avoids async/sync
                # generator exception propagation issues with ASGI.
                yield ("token", "[Error: The AI model is rate-limited (Tier 0: RPM=3, concurrent=1). Please wait 20-30 seconds and try again.]")
                return
            time.sleep(wait)

        except (APITimeoutError, APIConnectionError) as e:
            wait = 2 ** attempt * 5
            logger.warning(f"Stream connection issue (attempt {attempt}/{max_retries}), retrying in {wait}s: {e}")
            if attempt == max_retries:
                yield ("token", "[Error: Connection to AI model failed after retries. Please try again.]")
                return
            time.sleep(wait)
