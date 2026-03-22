# Summarization Service

Generates comprehensive investment analysis reports by extracting financial data from Supabase and feeding it to the Moonshot (Kimi) API.

## Architecture

```
Supabase (documents, earnings, price_snapshot)
    │
    ▼
fetcher.py  ──  Extract & organize data per ticker
    │
    ▼
prompts.py  ──  Build structured prompts
    │
    ▼
summarizer.py  ──  Call Moonshot API (OpenAI-compatible)
    │
    ▼
output/{TICKER}_{DATE}.md  ──  Markdown analysis reports
```

## Prerequisites

- Python 3.10+
- A Supabase project with `pgvector` enabled and populated tables (`documents`, `earnings`, `price_snapshot`, `tracked_tickers`). See `data-pipeline/pipeline/schema.sql` for the DDL.
- A Moonshot API key from [platform.moonshot.cn](https://platform.moonshot.cn)

## Installation

```bash
cd Summarization
pip install -r requirements.txt
```

## Configuration

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

| Variable | Description |
|---|---|
| `SUPABASE_DB_URL` | PostgreSQL connection string for Supabase |
| `MOONSHOT_API_KEY` | Moonshot (Kimi) API key |

Optional variables:

| Variable | Default | Description |
|---|---|---|
| `MOONSHOT_MODEL` | `moonshot-v1-128k` | Model to use (128k context recommended) |
| `MOONSHOT_BASE_URL` | `https://api.moonshot.cn/v1` | API endpoint |
| `MAX_CONTEXT_CHARS` | `380000` | Max chars for prompt context |

## Usage

```bash
# Single ticker
python run.py --ticker AAPL

# Multiple tickers
python run.py --ticker AAPL,MSFT,NVDA

# All active tickers from Supabase
python run.py --all

# Dry run — fetch data only, skip API call (useful for testing)
python run.py --ticker AAPL --dry-run
```

Reports are saved to `output/{TICKER}_{YYYY-MM-DD}.md`.

## Project Structure

```
Summarization/
├── run.py              # CLI entry point and orchestration
├── config.py           # Environment-driven configuration
├── fetcher.py          # Data extraction from Supabase + context assembly
├── prompts.py          # System and user prompt templates
├── summarizer.py       # Moonshot API client with retry logic
├── store/
│   ├── __init__.py
│   └── db.py           # PostgreSQL/Supabase data access layer
├── output/             # Generated reports (gitignored)
├── test-script/
│   └── testScriptForVector.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Data Flow

1. **Fetch** — `fetcher.py` queries Supabase for documents (news, 10-K, 10-Q, earnings), structured earnings rows, and price snapshots for a given ticker.
2. **Format** — Raw data is organized by type and formatted into readable text blocks. If total context exceeds the token budget, older news and filings are truncated.
3. **Prompt** — `prompts.py` assembles a system prompt (analyst role + report structure) and a user prompt (all formatted data).
4. **Generate** — `summarizer.py` sends the prompt to the Moonshot API with automatic retry on rate limits and connection errors.
5. **Save** — The generated report is written as a Markdown file to `output/`.

## Database Tables Used

| Table | Data Pulled |
|---|---|
| `documents` | News articles, SEC filings (10-K/10-Q), earnings text chunks |
| `earnings` | Quarterly EPS, revenue, net income, guidance |
| `price_snapshot` | Daily close price, P/E ratio, market cap |
| `tracked_tickers` | Active ticker list (used with `--all` flag) |
