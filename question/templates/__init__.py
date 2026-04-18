"""
question/templates
==================

方案 4 模板引擎 — 把"数字 + 叙述"解耦:
  1. LLM 只产 JSON 槽位 (schemas/*.py 定义 JSON Schema)
  2. 后端用本包的 Jinja2 沙盒环境渲染最终 Markdown

关键约束:
  * SandboxedEnvironment — 禁止访问 __class__ / __mro__ / 任何外部属性
  * StrictUndefined     — LLM 写错字段名立即抛错, 触发降级
  * 禁用 {% %} 块表达式 — 表格/循环由后端的 render_* 函数预先生成字符串
  * filters: money / pct / compact — 统一数字格式化
  * functions: yoy / qoq / sum_fy / pick_max / render_table
"""
from .renderer import render, RenderError

__all__ = ["render", "RenderError"]
