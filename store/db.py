# store/db.py
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    return psycopg2.connect(os.getenv("SUPABASE_DB_URL"))

def upsert_documents(docs: list[dict], embeddings: np.ndarray) -> int:
    conn = get_conn()
    cur = conn.cursor()
    values = []
    for i, doc in enumerate(docs):
        values.append((
            doc["id"],
            doc["content"],
            embeddings[i].tolist(),
            doc["ticker"],
            doc["date"],
            doc["source"],
            doc["doc_type"],
            doc.get("section"),
            doc.get("title"),
        ))
    execute_values(cur, """
        INSERT INTO documents (id, content, embedding, ticker, date, source, doc_type, section, title)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding
    """, values, template="(%s, %s, %s::vector, %s, %s, %s, %s, %s, %s)")
    count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return count

def upsert_earnings(ticker: str, quarter: str, date: str, eps: float = None,
                    revenue: int = None, net_income: int = None, guidance: str = None) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO earnings (ticker, quarter, date, eps, revenue, net_income, guidance)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, quarter) DO UPDATE SET
            eps = EXCLUDED.eps,
            revenue = EXCLUDED.revenue,
            net_income = EXCLUDED.net_income,
            guidance = EXCLUDED.guidance
    """, (ticker, quarter, date, eps, revenue, net_income, guidance))
    conn.commit()
    cur.close()
    conn.close()

def semantic_search(query_embedding: list[float], tickers: list[str], top_k: int = 5) -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, content, ticker, date, source, doc_type, section, title,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        WHERE ticker = ANY(%s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, tickers, query_embedding, top_k))
    columns = [desc[0] for desc in cur.description]
    results = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results

def get_earnings(ticker: str, limit: int = 4) -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT ticker, quarter, date, eps, revenue, net_income, guidance
        FROM earnings
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT %s
    """, (ticker, limit))
    columns = [desc[0] for desc in cur.description]
    results = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results


def get_documents_by_ticker(
    ticker: str,
    doc_type: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """Fetch documents for a ticker, ordered by date DESC.
    Skips the embedding column to keep memory usage low.
    """
    conn = get_conn()
    cur = conn.cursor()
    if doc_type:
        cur.execute("""
            SELECT id, content, ticker, date, source, doc_type, section, title
            FROM documents
            WHERE ticker = %s AND doc_type = %s
            ORDER BY date DESC
            LIMIT %s
        """, (ticker, doc_type, limit))
    else:
        cur.execute("""
            SELECT id, content, ticker, date, source, doc_type, section, title
            FROM documents
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT %s
        """, (ticker, limit))
    columns = [desc[0] for desc in cur.description]
    results = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results


def get_price_snapshots(ticker: str, limit: int = 30) -> list[dict]:
    """Fetch recent price snapshots for a ticker, ordered by date DESC."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT ticker, date, close_price, pe_ratio, market_cap
        FROM price_snapshot
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT %s
    """, (ticker, limit))
    columns = [desc[0] for desc in cur.description]
    results = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results


def get_tracked_tickers(active_only: bool = True) -> list[dict]:
    """Fetch tracked tickers from Supabase."""
    conn = get_conn()
    cur = conn.cursor()
    if active_only:
        cur.execute("""
            SELECT ticker, ticker_type
            FROM tracked_tickers
            WHERE is_active = TRUE
            ORDER BY ticker
        """)
    else:
        cur.execute("""
            SELECT ticker, ticker_type
            FROM tracked_tickers
            ORDER BY ticker
        """)
    columns = [desc[0] for desc in cur.description]
    results = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results