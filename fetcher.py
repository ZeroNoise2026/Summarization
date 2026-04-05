"""
fetcher.py
Extracts data from Supabase and assembles structured context for each ticker.
"""

import logging
from dataclasses import dataclass, field

from store.db import (
    get_documents_by_ticker,
    get_earnings,
    get_price_snapshots,
)
from config import MAX_CONTEXT_CHARS

logger = logging.getLogger(__name__)


@dataclass
class TickerContext:
    ticker: str
    news_text: str = ""
    filings_text: str = ""
    earnings_text: str = ""
    price_text: str = ""
    doc_counts: dict = field(default_factory=dict)

    @property
    def total_chars(self) -> int:
        return len(self.news_text) + len(self.filings_text) + len(self.earnings_text) + len(self.price_text)


def _format_news(docs: list[dict]) -> str:
    if not docs:
        return ""
    lines = []
    for d in docs:
        date = d.get("date", "N/A")
        title = d.get("title") or ""
        content = d.get("content", "")
        header = f"[{date}] {title}".strip() if title else f"[{date}]"
        lines.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(lines)


def _format_filings(docs: list[dict]) -> str:
    if not docs:
        return ""
    lines = []
    for d in docs:
        date = d.get("date", "N/A")
        doc_type = d.get("doc_type", "filing")
        source = d.get("source", "")
        section = d.get("section") or ""
        content = d.get("content", "")
        header = f"[{date}] {doc_type} ({source})"
        if section:
            header += f" - {section}"
        lines.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(lines)


def _format_earnings(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = ["Quarter | Date | EPS | Revenue | Net Income | Guidance"]
    lines.append("---|---|---|---|---|---")
    for r in rows:
        eps = f"{r['eps']:.2f}" if r.get("eps") is not None else "N/A"
        rev = f"${r['revenue']:,}" if r.get("revenue") is not None else "N/A"
        ni = f"${r['net_income']:,}" if r.get("net_income") is not None else "N/A"
        guidance = r.get("guidance") or "N/A"
        lines.append(f"{r['quarter']} | {r.get('date', 'N/A')} | {eps} | {rev} | {ni} | {guidance}")
    return "\n".join(lines)


def _format_prices(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = ["Date | Close | P/E | Market Cap"]
    lines.append("---|---|---|---")
    for r in rows:
        close = f"${r['close_price']:.2f}" if r.get("close_price") is not None else "N/A"
        pe = f"{r['pe_ratio']:.1f}" if r.get("pe_ratio") is not None else "N/A"
        mc = f"${r['market_cap']:,}" if r.get("market_cap") is not None else "N/A"
        lines.append(f"{r.get('date', 'N/A')} | {close} | {pe} | {mc}")
    return "\n".join(lines)


def _truncate_to_budget(ctx: TickerContext, budget: int) -> TickerContext:
    """If total context exceeds budget, trim news first (most voluminous),
    then filings, keeping earnings and prices intact (small & structured)."""
    if ctx.total_chars <= budget:
        return ctx

    fixed = len(ctx.earnings_text) + len(ctx.price_text)
    remaining = budget - fixed

    filings_budget = min(len(ctx.filings_text), remaining // 2)
    news_budget = remaining - filings_budget

    if len(ctx.news_text) > news_budget:
        ctx.news_text = ctx.news_text[:news_budget] + "\n\n... [truncated due to length]"
    if len(ctx.filings_text) > filings_budget:
        ctx.filings_text = ctx.filings_text[:filings_budget] + "\n\n... [truncated due to length]"

    return ctx


def fetch_context(ticker: str) -> TickerContext:
    """Fetch all relevant data from Supabase and assemble into TickerContext."""
    logger.info(f"Fetching data for {ticker}...")

    news_docs = get_documents_by_ticker(ticker, doc_type="news", limit=100)
    filing_docs = (
        get_documents_by_ticker(ticker, doc_type="10-K", limit=50)
        + get_documents_by_ticker(ticker, doc_type="10-Q", limit=50)
    )
    earnings_docs = get_documents_by_ticker(ticker, doc_type="earnings", limit=50)
    earnings_rows = get_earnings(ticker, limit=20)
    price_rows = get_price_snapshots(ticker, limit=30)

    all_filing_docs = filing_docs + earnings_docs

    ctx = TickerContext(
        ticker=ticker,
        news_text=_format_news(news_docs),
        filings_text=_format_filings(all_filing_docs),
        earnings_text=_format_earnings(earnings_rows),
        price_text=_format_prices(price_rows),
        doc_counts={
            "news": len(news_docs),
            "filings": len(filing_docs),
            "earnings_docs": len(earnings_docs),
            "earnings_rows": len(earnings_rows),
            "prices": len(price_rows),
        },
    )

    ctx = _truncate_to_budget(ctx, MAX_CONTEXT_CHARS)

    logger.info(
        f"  {ticker}: {ctx.doc_counts} | context chars: {ctx.total_chars:,}"
    )
    return ctx
