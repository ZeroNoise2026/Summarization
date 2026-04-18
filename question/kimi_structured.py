"""
question/kimi_structured.py
方案 4 — 调用 KIMI 产出结构化 JSON, 校验后返回.

与 shared/llm.py::chat() 的区别:
  * 强制 response_format={"type":"json_object"}
  * jsonschema 验证, 失败重试 1 次
  * 返回 dict (已 parse), 失败抛 StructuredGenError

⚠️ moonshot 对 json_object 支持: 文档要求 prompt 里包含 "JSON" 字样.
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any

from jsonschema import Draft202012Validator, ValidationError
from openai import RateLimitError, APITimeoutError, APIConnectionError

from shared.llm import get_client, pick_model

logger = logging.getLogger(__name__)


class StructuredGenError(RuntimeError):
    """LLM 未能产出合法 JSON. 调用方应 fallback."""


def generate_json(
    *,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    few_shot: list[dict[str, str]] | None = None,
    intent_label: str = "unknown",
    max_retries: int = 2,
    temperature: float = 0.3,
    max_tokens: int = 1200,
) -> tuple[dict[str, Any], dict]:
    """Call KIMI in JSON mode, validate against schema. Returns (parsed_json, usage).

    Raises StructuredGenError on repeated failure — caller must fallback.
    """
    client = get_client()
    validator = Draft202012Validator(schema)

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if few_shot:
        messages.extend(few_shot)
    messages.append({"role": "user", "content": user_prompt})

    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    model = pick_model(total_chars)
    # json_object mode: kimi-k2.5 is reasoning-only and may not honour this cleanly;
    # prefer non-reasoning tier for structured output.
    if "kimi-k2" in model:
        model = "moonshot-v1-32k"

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            usage = resp.usage
            prompt_toks = getattr(usage, "prompt_tokens", 0) or 0
            completion_toks = getattr(usage, "completion_tokens", 0) or 0
            total_toks = getattr(usage, "total_tokens", 0) or 0
            latency_ms = int((time.time() - t0) * 1000)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as je:
                raise StructuredGenError(f"Invalid JSON from LLM: {je}; raw[:200]={raw[:200]!r}")

            # schema validate
            errors = sorted(validator.iter_errors(parsed), key=lambda e: e.path)
            if errors:
                msg = "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:3])
                raise StructuredGenError(f"Schema violation: {msg}")

            # audit
            try:
                from audit import log_api_call
                log_api_call(
                    service="question",
                    api_type="kimi_llm",
                    endpoint="chat.completions.structured",
                    model=model,
                    tokens_in=prompt_toks,
                    tokens_out=completion_toks,
                    tokens_total=total_toks,
                    latency_ms=latency_ms,
                )
            except Exception:
                pass

            logger.info(
                f"structured_gen OK: intent={intent_label} model={model} "
                f"tokens={total_toks} latency={latency_ms}ms attempt={attempt}"
            )
            return parsed, {
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "total_tokens": total_toks,
                "latency_ms": latency_ms,
                "model": model,
            }

        except StructuredGenError as e:
            last_err = e
            logger.warning(f"structured_gen retry {attempt}/{max_retries}: {e}")
            # Add a corrective hint for next attempt
            messages = messages + [
                {"role": "user", "content": (
                    "Your previous response was not valid JSON matching the required schema. "
                    f"Error: {e}. Reply again with ONLY the JSON object, no prose."
                )}
            ]
        except RateLimitError as e:
            last_err = e
            wait = min(2 ** attempt * 5, 30)
            logger.warning(f"structured_gen rate limited, waiting {wait}s: {e}")
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as e:
            last_err = e
            logger.warning(f"structured_gen conn error (attempt {attempt}): {e}")
            time.sleep(2)
        except Exception as e:
            last_err = e
            logger.error(f"structured_gen unexpected error: {type(e).__name__}: {e}")
            break

    raise StructuredGenError(f"Failed after {max_retries} attempts: {last_err}")
