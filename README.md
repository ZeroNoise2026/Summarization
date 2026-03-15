# VectorDB Storage

This repository provides a python utility for storing and managing unstructured text documents along with their vector embeddings, as well as structured financial earnings data, within a PostgreSQL database (like Supabase configured with `pgvector`).

## Features
- **Document & Embedding Storage**: Store text data (e.g., news, SEC filings) coupled with high-dimensional embedding vectors using `pgvector`.
- **Earnings Data Storage**: Keep track of quarterly financial reports, revenue, EPS (Earnings Per Share), and net income guidance.
- **Upsert Functionality**: Handles duplicate records gracefully by updating existing entries on conflict.

## Installation

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Configuration

This project requires a connection string to a PostgreSQL database (e.g., Supabase) with the `pgvector` extension enabled. 

Create a `.env` file in the root directory and add your connection URL:

```env
SUPABASE_DB_URL="postgresql://user:password@host:port/dbname"
```

## Usage

### Storing Documents and Embeddings

You can use the `upsert_documents` function in `store/db.py` to save documents alongside their vector representations. 

Example (`test-script/testScriptForVector.py`):
```python
import numpy as np
from store.db import upsert_documents

fake_doc = {
    "id": "test_001",
    "content": "Apple reports record Q1 earnings",
    "ticker": "AAPL",
    "date": "2026-03-15",
    "source": "finnhub",
    "doc_type": "news",
    "section": None,
    "title": "Apple Q1 Earnings"
}

# Generate a 384-dimensional fake embedding vector
fake_embedding = np.random.rand(1, 384).astype(np.float32)

# Insert or update the document
count = upsert_documents([fake_doc], fake_embedding)
print(f"Inserted: {count} document(s)")
```

### Storing Earnings Data

You can use `upsert_earnings` to save structured financial data:

```python
from store.db import upsert_earnings

upsert_earnings(
    ticker="AAPL",
    quarter="Q1 2026",
    date="2026-03-15",
    eps=1.20,
    revenue=120000000000,
    net_income=30000000000,
    guidance="Strong growth expected in Q2."
)
```

## Database Schema Dependencies

To use this code out of the box, the following minimum PostgreSQL table structures are required.

**`documents`** table:
- `id` (Primary Key)
- `content` (Text)
- `embedding` (Vector type)
- `ticker` (String)
- `date` (Date/String)
- `source` (String)
- `doc_type` (String)
- `section` (String, Optional)
- `title` (String, Optional)

**`earnings`** table:
- `ticker` (Component of Primary Key)
- `quarter` (Component of Primary Key)
- `date` (Date/String)
- `eps` (Float)
- `revenue` (BigInt)
- `net_income` (BigInt)
- `guidance` (Text, Optional)
