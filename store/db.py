# store/db.py
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import os
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