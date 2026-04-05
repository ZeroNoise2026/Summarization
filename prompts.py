"""
prompts.py
Prompt templates for generating investment analysis reports via Moonshot (Kimi).
"""

from fetcher import TickerContext

SYSTEM_PROMPT = """You are a senior financial analyst skilled at synthesizing multi-source financial data into concise investment summaries.

Your summary must meet the following requirements:
1. STRICT 300-word limit — be concise and prioritize the most important information
2. Well-structured, using Markdown format
3. All data citations must be specific (dates, numbers) — never fabricate data
4. If a category of data is missing, skip it — do not invent information

Cover these areas briefly:
- Company overview & recent developments
- Stock price & valuation
- Financial highlights (revenue, EPS)
- Key news & risks
- Outlook"""


def build_user_prompt(ctx: TickerContext) -> str:
    """Assemble all context into a single user prompt for the LLM."""
    sections = [f"Based on the following data, generate a comprehensive investment analysis report for **{ctx.ticker}**.\n"]

    if ctx.price_text:
        sections.append(f"### Recent Price Data\n\n{ctx.price_text}")

    if ctx.earnings_text:
        sections.append(f"### Earnings Data\n\n{ctx.earnings_text}")

    if ctx.news_text:
        sections.append(f"### News & Developments\n\n{ctx.news_text}")

    if ctx.filings_text:
        sections.append(f"### SEC Filings / Financial Report Content\n\n{ctx.filings_text}")

    if not any([ctx.price_text, ctx.earnings_text, ctx.news_text, ctx.filings_text]):
        sections.append("(No data available)")

    return "\n\n".join(sections)
