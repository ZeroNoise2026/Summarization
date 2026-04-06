"""
summary/fetcher.py
Extracts data from Supabase and assembles structured context for each ticker.
格式化函数已提取到 shared/formatters.py, 这里调用共享版本。
"""

import logging
from dataclasses import dataclass, field

from shared.db import (
    get_documents_by_ticker,
    get_earnings,
    get_price_snapshots,
)
from shared.formatters import (
    format_news,
    format_filings,
    format_earnings,
    format_prices,
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


def _truncate_to_budget(ctx: TickerContext, budget: int) -> TickerContext:
    """If total context exceeds budget, trim news first, then filings."""
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
        news_text=format_news(news_docs),
        filings_text=format_filings(all_filing_docs),
        earnings_text=format_earnings(earnings_rows),
        price_text=format_prices(price_rows),
        doc_counts={
            "news": len(news_docs),
            "filings": len(filing_docs),
            "earnings_docs": len(earnings_docs),
            "earnings_rows": len(earnings_rows),
            "prices": len(price_rows),
        },
    )
    ctx = _truncate_to_budget(ctx, MAX_CONTEXT_CHARS)
    logger.info(f"  {ticker}: {ctx.doc_counts} | context chars: {ctx.total_chars:,}")
    return ctx
