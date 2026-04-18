"""
question/templates/engine.py
Jinja2 SandboxedEnvironment — 单例, 启动时构建一次.

安全策略:
  * 禁用 {% block %} / {% for %} / {% if %} 等块语法 — LLM 只能用 {{var}} 和 | filter
    实现方式: 重写 block_start_string/block_end_string 为不可能在正常文本出现的哨兵串,
    任何 {% ... %} 都会被当作普通文本. 这比扫描字符串做正则更可靠.
  * StrictUndefined — 模板里出现未定义变量时抛 UndefinedError (不静默输出空串).
  * autoescape=False — 我们输出 Markdown, 不是 HTML.
"""
from __future__ import annotations
import logging
from jinja2.sandbox import SandboxedEnvironment
from jinja2 import StrictUndefined, Template

from .filters import ALL_FILTERS
from .functions import ALL_FUNCTIONS

logger = logging.getLogger(__name__)

# 哨兵串: 32 字节随机文本, 正文里出现概率 ≈ 0. 等效于"禁用块语法".
_BLOCK_SENTINEL_START = "{%__QA_PLAN4_BLOCK_DISABLED__"
_BLOCK_SENTINEL_END = "__QA_PLAN4_BLOCK_DISABLED__%}"

_env: SandboxedEnvironment | None = None


def get_env() -> SandboxedEnvironment:
    """Lazy singleton — 全进程共享一个 Environment (模板编译缓存)."""
    global _env
    if _env is None:
        env = SandboxedEnvironment(
            block_start_string=_BLOCK_SENTINEL_START,
            block_end_string=_BLOCK_SENTINEL_END,
            variable_start_string="{{",
            variable_end_string="}}",
            comment_start_string="{#",
            comment_end_string="#}",
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=False,
            lstrip_blocks=False,
        )
        for name, fn in ALL_FILTERS.items():
            env.filters[name] = fn
        for name, fn in ALL_FUNCTIONS.items():
            env.globals[name] = fn
        _env = env
        logger.info(
            f"Jinja2 sandbox initialized: filters={list(ALL_FILTERS)}, "
            f"globals={list(ALL_FUNCTIONS)}, blocks=DISABLED, strict=True"
        )
    return _env


def compile_template(source: str) -> Template:
    """Compile a template string under the sandboxed env."""
    return get_env().from_string(source)
