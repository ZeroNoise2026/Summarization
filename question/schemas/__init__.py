"""
question/schemas
================
方案 4 — JSON Schema + 模板定义. 每个 intent 一个文件, 导出:

  SCHEMA          — jsonschema dict, 用来校验 LLM 输出 (EARNINGS / COMPARISON 用)
  TEMPLATE        — Jinja2 模板字符串
  build_context() — 从 DB 数据构造 render context (注入后端预算好的 yoy/qoq/table 等)
  SYSTEM_PROMPT   — (仅 LLM 参与的 intent) 告诉 KIMI 按 schema 输出 JSON
  FEW_SHOT        — (同上) 示例对
"""
