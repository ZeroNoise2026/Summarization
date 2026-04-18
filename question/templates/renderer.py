"""
question/templates/renderer.py
主入口 — 把 (template_source, context) 渲染成最终 Markdown 字符串.

任何异常 (未定义变量 / 渲染错) 都包成 RenderError, 调用方捕获后走 fallback.
"""
from __future__ import annotations
import logging
from typing import Any
from jinja2 import UndefinedError, TemplateSyntaxError

from .engine import compile_template

logger = logging.getLogger(__name__)


class RenderError(RuntimeError):
    """模板渲染失败 — 调用方应 fallback 到原生成路径."""


def render(template_source: str, context: dict[str, Any]) -> str:
    """Render template string with given context. Raises RenderError on any failure.

    Also validates that no literal "{{" or "}}" slipped through (indicates the LLM
    produced a field name that was interpreted as text rather than a real variable,
    which StrictUndefined should already catch — this is belt-and-braces).
    """
    try:
        tpl = compile_template(template_source)
        out = tpl.render(**context)
    except UndefinedError as e:
        raise RenderError(f"Undefined variable during render: {e}") from e
    except TemplateSyntaxError as e:
        raise RenderError(f"Template syntax error: {e}") from e
    except Exception as e:  # pragma: no cover — 捕获一切防崩溃
        raise RenderError(f"Unexpected render error: {type(e).__name__}: {e}") from e

    # 防御: 渲染完输出里不应再有 {{ }} 残留
    if "{{" in out or "}}" in out:
        raise RenderError(f"Residual mustache markers in output (first 120 chars): {out[:120]!r}")
    return out
