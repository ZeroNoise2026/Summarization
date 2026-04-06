"""
question/retriever.py
Data retrieval layer — Tier 1 (DB semantic search + structured queries).
"""

import logging
import time
from typing import Optional

from config import SEMANTIC_TOP_K, SIMILARITY_THRESHOLD
from question.models import Intent, QueryMode, Source
from shared.db import (
    get_documents_by_embedding,
    get_earnings,
    get_price_snapshots,
    get_tracked_tickers,
)
from shared.formatters import (
    format_news,
    format_filings,
    format_earnings,
    format_prices,
)
from shared.http import get_embedding_client

logger = logging.getLogger(__name__)


async def _encode_query(query: str) -> list[float]:
    """Call embedding-service to convert query text into a 384-dim vector."""
    t0 = time.time()
    client = get_embedding_client()
    resp = await client.post(
        "/api/encode-query",
        json={"query": query},
    )
    resp.raise_for_status()
    latency = int((time.time() - t0) * 1000)

    from audit import log_api_call
    log_api_call(
        service="question",
        api_type="embedding",
        endpoint="/api/encode-query",
        latency_ms=latency,
    )
    return resp.json()["embedding"]


async def semantic_search(
    query: str,
    ticker: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> tuple[list[dict], list[Source]]:
    """Semantic search: query -> embedding -> pgvector -> return docs + sources."""
    embedding = await _encode_query(query)
    docs = get_documents_by_embedding(
        query_embedding=embedding,
        ticker=ticker,
        doc_type=doc_type,
        top_k=SEMANTIC_TOP_K,
        threshold=SIMILARITY_THRESHOLD,
    )

    sources = []
    for d in docs:
        sources.append(Source(
            doc_type=d.get("doc_type", "news"),
            ticker=d.get("ticker", ""),
            date=d.get("date"),
            title=d.get("title"),
            similarity=d.get("similarity"),
        ))

    return docs, sources


def structured_lookup(
    ticker: str,
    intent: Intent,
) -> tuple[str, list[Source]]:
    """Structured query: fetch data directly from DB tables (earnings, price_snapshot)."""
    sources: list[Source] = []
    parts: list[str] = []

    if intent in (Intent.EARNINGS_ANALYSIS, Intent.TRADE_SUGGESTION, Intent.COMPARISON, Intent.GENERAL_ANALYSIS):
        rows = get_earnings(ticker, limit=8)
        if rows:
            parts.append(f"### {ticker} Earnings\n\n{format_earnings(rows)}")
            sources.append(Source(doc_type="earnings", ticker=ticker, date=rows[0].get("date")))

    if intent in (Intent.PRICE_QUERY, Intent.TRADE_SUGGESTION, Intent.COMPARISON, Intent.GENERAL_ANALYSIS):
        rows = get_price_snapshots(ticker, limit=5)
        if rows:
            parts.append(f"### {ticker} Price Snapshot\n\n{format_prices(rows)}")
            sources.append(Source(doc_type="price", ticker=ticker, date=rows[0].get("date")))

    return "\n\n".join(parts), sources


async def retrieve(
    query: str,
    tickers: list[str],
    mode: QueryMode,
    intent: Intent,
) -> tuple[str, list[Source], str]:
    """Main retrieval entry point. Dispatches to semantic / structured / hybrid based on mode.

    Returns:
        (context_text, sources, data_freshness)
    """
    all_parts: list[str] = []
    all_sources: list[Source] = []
    data_freshness = ""

    # Get data freshness from tracked_tickers
    tracked = get_tracked_tickers(active_only=True)
    tracked_map = {r["ticker"]: r for r in tracked}
    for t in tickers:
        info = tracked_map.get(t)
        if info and info.get("last_successful_run"):
            data_freshness = info["last_successful_run"]
            break

    # Semantic search
    if mode in (QueryMode.SEMANTIC, QueryMode.HYBRID):
        for ticker in tickers:
            doc_type = None
            if intent == Intent.NEWS_SUMMARY:
                doc_type = "news"
            docs, sources = await semantic_search(query, ticker=ticker, doc_type=doc_type)
            if docs:
                text_parts = []
                for d in docs:
                    header = f"[{d.get('date', 'N/A')}] {d.get('title', '')}".strip()
                    text_parts.append(f"{header}\n{d.get('content', '')}")
                all_parts.append(f"### {ticker} Related Documents\n\n" + "\n\n---\n\n".join(text_parts))
                all_sources.extend(sources)

    # Structured query
    if mode in (QueryMode.STRUCTURED, QueryMode.HYBRID):
        for ticker in tickers:
            text, sources = structured_lookup(ticker, intent)
            if text:
                all_parts.append(text)
                all_sources.extend(sources)

    context = "\n\n".join(all_parts) if all_parts else "(No relevant data found)"
    return context, all_sources, data_freshness
