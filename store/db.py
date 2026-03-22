# store/db.py
# Supabase data access layer using REST API (supabase-py).

import logging
from typing import Optional

from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

_client: Optional[Client] = None

FIELDS_DOCUMENTS = "id, content, ticker, date, source, doc_type, section, title"


def _get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def get_documents_by_ticker(
    ticker: str,
    doc_type: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """Fetch documents for a ticker, ordered by date DESC."""
    client = _get_client()
    query = (
        client.table("documents")
        .select(FIELDS_DOCUMENTS)
        .eq("ticker", ticker)
        .order("date", desc=True)
        .limit(limit)
    )
    if doc_type:
        query = query.eq("doc_type", doc_type)
    return query.execute().data


def get_earnings(ticker: str, limit: int = 20) -> list[dict]:
    """Fetch earnings rows for a ticker, ordered by date DESC."""
    client = _get_client()
    return (
        client.table("earnings")
        .select("ticker, quarter, date, eps, revenue, net_income, guidance")
        .eq("ticker", ticker)
        .order("date", desc=True)
        .limit(limit)
        .execute()
        .data
    )


def get_price_snapshots(ticker: str, limit: int = 30) -> list[dict]:
    """Fetch recent price snapshots for a ticker, ordered by date DESC."""
    client = _get_client()
    return (
        client.table("price_snapshot")
        .select("ticker, date, close_price, pe_ratio, market_cap")
        .eq("ticker", ticker)
        .order("date", desc=True)
        .limit(limit)
        .execute()
        .data
    )


def get_tracked_tickers(active_only: bool = True) -> list[dict]:
    """Fetch tracked tickers from Supabase."""
    client = _get_client()
    query = (
        client.table("tracked_tickers")
        .select("ticker, ticker_type")
        .order("ticker")
    )
    if active_only:
        query = query.eq("is_active", True)
    return query.execute().data
