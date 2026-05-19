# Summarization

Two co-located services sharing the same data layer and LLM client:

| Service | Form | Purpose |
|---------|------|---------|
| **summary** | CLI / scheduled job | Generates a comprehensive Markdown research report per ticker by pulling everything from Supabase and prompting Moonshot. Caches by input hash so unchanged inputs reuse a previous report. |
| **question** | FastAPI service (port 8003) | Powers RAG-based ad-hoc Q&A: structured-output intents (earnings, comparison, price) plus a templated rendering layer that keeps numbers deterministic and lets the LLM write only the prose. |

> **For PMs / analysts:** This repo is where the "deep analysis" lives.
> `summary` is the engine behind the daily research report you can pull
> for any ticker on your watchlist. `question` is the engine behind the
> chat answers in ChatbotUI when you ask questions like "how was AAPL's
> last quarter" — it makes sure the numbers in the answer come from the
> database, not from the model's memory.

> **For engineers:** Two services, one repo, one venv, one config.
> `summary/` is a CLI; `question/` is a FastAPI app. They share
> `config.py`, `shared/` (LLM client, DB access, formatters), and
> `audit/` (logging). Both are also consumers of the top-level
> `skills/` framework (see [Skills integration](#skills-integration)
> below).

## Architecture

```
                Supabase (documents, earnings, price_snapshot, summary_cache)
                                  ▲          ▲
                                  │          │
       ┌──────────────────────────┘          └──────────────────────────┐
       │                                                                │
   summary/                                                        question/
   ┌──────────────────────┐                                    ┌──────────────────────┐
   │ fetcher → prompts →  │                                    │ FastAPI :8003        │
   │ summarizer (Moonshot)│                                    │  /api/ask/stream     │
   │ → cache → output/    │                                    │ router → orchestrator│
   └──────────────────────┘                                    │   ├─ retriever (RAG) │
                                                               │   ├─ live_fetcher    │
                                                               │   ├─ kimi_structured │
                                                               │   └─ templates/      │
                                                               └──────────────────────┘
```

## Layout

```
Summarization/
├── config.py                    single source of truth for env vars (both services)
├── requirements.txt
├── Dockerfile.summary           image for the CLI / scheduled job
├── Dockerfile.question          image for the FastAPI service
│
├── summary/                     ── offline report service ──
│   ├── run.py                   CLI entry point (python -m summary.run)
│   ├── fetcher.py               assemble TickerContext from Supabase
│   ├── prompts.py               SYSTEM_PROMPT, build_user_prompt, PROMPT_VERSION
│   ├── summarizer.py            Moonshot call wrapper
│   ├── cache.py                 summary_cache table integration (input_hash)
│   └── cleanup.py               retention sweep for old reports
│
├── question/                    ── online Q&A service ──
│   ├── main.py                  FastAPI app (port 8003)
│   ├── orchestrator.py          per-request flow
│   ├── router.py                intent classification (rule + LLM)
│   ├── retriever.py             pgvector search w/ recency reranking
│   ├── live_fetcher.py          real-time price / quote fallback
│   ├── kimi_structured.py       JSON-schema-constrained LLM calls
│   ├── tier2_cache.py           in-process LRU for hot tickers
│   ├── generator.py             free-form answer path (when no structured intent fits)
│   ├── schemas/                 per-intent JSON Schema + Jinja2 templates
│   │   ├── earnings_analysis.py
│   │   ├── comparison.py
│   │   └── price_query.py
│   └── templates/               sandboxed Jinja2 engine + filters/functions
│       ├── engine.py            SandboxedEnvironment, blocks disabled
│       ├── filters.py           money / pct / compact formatters
│       ├── functions.py         YoY / QoQ / FY aggregations (backend-computed)
│       └── renderer.py          template entry point + RenderError handling
│
├── shared/                      cross-service helpers
│   ├── db.py                    Supabase / pgvector access
│   ├── llm.py                   Moonshot client + model tier selection + retry
│   ├── http.py                  outbound HTTP helpers
│   └── formatters.py            text formatters for context building
│
├── audit/                       structured logging to Supabase
├── scripts/                     SQL / setup / dev tools (incl. validate_supabase.py)
├── output/                      generated reports {TICKER}_{YYYY-MM-DD}.md (gitignored)
├── tests/                       unit + integration tests for both services
└── test-script/                 ad-hoc scripts (vector probes, etc.)
```

## Prerequisites

- Python 3.11+
- Supabase project with the schema in `data-pipeline/pipeline/schema.sql`
  applied, and `pgvector` enabled.
- Moonshot API key from [platform.moonshot.cn](https://platform.moonshot.cn).
- A reachable embedding-service (port 8002) — `question/` calls it to
  encode user queries before vector search.

## Setup

```bash
cd Summarization
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then fill in credentials
```

## summary — CLI usage

```bash
# Single ticker
python -m summary.run --ticker AAPL

# Multiple tickers (comma-separated)
python -m summary.run --ticker AAPL,MSFT,NVDA

# All active tickers from Supabase (the production cron path)
python -m summary.run --all

# Fetch + assemble context but skip the LLM call (no cost, for sanity checks)
python -m summary.run --ticker AAPL --dry-run

# List skills available via the skills framework
python -m summary.run --list-skills

# Invoke a skill explicitly
python -m summary.run --skill generate_report --args ticker=AAPL
python -m summary.run --skill compare_tickers \
    --args '{"tickers":["AAPL","MSFT"]}'
```

Reports land under `output/{TICKER}_{YYYY-MM-DD}.md`. Cache lookups are
keyed on `SHA256(sorted source doc ids + model + prompt_version)` — if
nothing changed, no LLM call is made.

## question — running the service

```bash
# Local dev (port 8003)
uvicorn question.main:app --port 8003 --reload

# Docker (Cloud Run-ready; honors $PORT, defaults to 8080)
docker build -f Dockerfile.question -t qa-question .
docker run -p 8003:8080 --env-file .env qa-question
```

Endpoint: `POST /api/ask/stream` — SSE stream of tokens + structured
events (sources, intents, etc.). Consumed by `ChatbotUI/backend/rag.py`.

## Skills integration

Both services are consumers of the top-level `skills/` framework
(see [`../skills/README.md`](../skills/README.md)):

- **`summary/run.py`** supports `--skill <name>` to route through the
  framework instead of the legacy `--ticker` path. The legacy path is
  preserved verbatim for backward compatibility.
- **`question/`** does not currently call skills directly; it remains
  the RAG/structured-intent service. Chat-side skill routing lives in
  `ChatbotUI/backend/main.py`, which has its own router → skill →
  fallback-to-RAG handoff.

The shipped skills (`generate_report`, `compare_tickers`) call back
into this repo's `summary.fetcher` and `summary.summarizer` for data
and LLM access, so behavior between "skill in chat" and "CLI report"
stays in lockstep.

## Configuration

All env vars live in `config.py` (single source of truth). Required:

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL`, `SUPABASE_KEY` | Supabase credentials |
| `MOONSHOT_API_KEY` | Moonshot API key (used by both services + skills) |
| `EMBEDDING_SERVICE_URL` | URL of embedding-service (e.g. `http://localhost:8002`) |

Common optional tunables (see `config.py` for the full list):

| Variable | Default | Purpose |
|----------|---------|---------|
| `MOONSHOT_MODEL` | `kimi-k2.5` | Report model (summary service) |
| `MAX_CONTEXT_CHARS` | `380000` | Per-ticker context budget for reports |
| `SEMANTIC_TOP_K` | `8` | Retriever top-K |
| `SIMILARITY_THRESHOLD` | `0.3` | Cosine threshold |
| `RECENCY_HALF_LIFE_DAYS` | `14` | Recency-rerank Gaussian half-life |
| `TIER2_CACHE_MAX_SIZE` | `200` | LRU size for question-service hot data |

## Data flow (summary service)

1. **Fetch** — `fetcher.fetch_context(ticker)` queries Supabase for
   documents (news, 10-K, 10-Q, earnings text), structured earnings
   rows, and price snapshots.
2. **Format** — Raw data is organized by type into a `TickerContext`.
   If total context exceeds `MAX_CONTEXT_CHARS`, news and filings are
   truncated first (earnings and price stay intact).
3. **Cache check** — `compute_input_hash` over sorted source doc IDs +
   model + prompt version. Cache hit → reuse stored report.
4. **Prompt** — `prompts.SYSTEM_PROMPT` + `build_user_prompt(ctx)`.
5. **Generate** — `summarizer.generate_summary` calls Moonshot with
   retry on rate-limit / connection errors.
6. **Save** — Report is written to `output/` and persisted to
   `summary_cache` keyed on the input hash.

## Data flow (question service)

Per request to `/api/ask/stream`:

1. **Route** — `router.classify()` picks an Intent (`EARNINGS_ANALYSIS`,
   `COMPARISON`, `PRICE_QUERY`, `NEWS_SUMMARY`, ...). Uses keyword rules
   first, then an LLM fallback for ambiguous queries.
2. **Retrieve** — For RAG paths: encode query via embedding-service,
   vector-search Supabase with recency-aware reranking.
3. **Structured generation (when applicable)** — For
   `EARNINGS_ANALYSIS` / `COMPARISON` / `PRICE_QUERY`, build context
   deterministically from DB rows, ask Moonshot for only the
   narrative slots (`json_object` mode), then render the final Markdown
   through the sandboxed Jinja2 templates. Numbers are computed in
   `templates/functions.py` (YoY / QoQ / FY aggregations), not by the
   LLM.
4. **Free-form fallback** — For intents without a structured template,
   `generator.py` produces a normal answer from retrieved docs.
5. **Stream** — Tokens and structured events (`sources`, `intent`) are
   emitted as SSE.

## Database tables used

| Table | Used by | Purpose |
|-------|---------|---------|
| `documents` | both | News, SEC filings (10-K/10-Q), earnings text chunks with embeddings |
| `earnings` | both | Quarterly EPS, revenue, net income, guidance |
| `price_snapshot` | both | Daily close, P/E, market cap |
| `tracked_tickers` | summary | Active ticker list (used with `--all`) |
| `summary_cache` | summary | Cached generated reports keyed on input_hash |
| `audit_log` | both | Structured query/response logging |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Just the templates layer (fast)
pytest tests/test_templates_phase1.py -v

# Smoke test for vector retrieval (needs Supabase reachable)
python scripts/test_retrieval.py
```

## Related services

| Service | Why this repo cares |
|---------|---------------------|
| `embedding-service` (:8002) | Called by `question/` to encode user queries before pgvector search. |
| `data-pipeline` | Populates the Supabase tables this repo reads from. We never write to those tables from here. |
| `ChatbotUI/backend` | Calls `question/` via HTTP and embeds skill execution into its chat stream. |
| `skills/` (top-level) | Hosts `generate_report` and `compare_tickers`, which reuse this repo's fetcher + summarizer. |
