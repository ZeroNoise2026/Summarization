"""
question/orchestrator.py
Main orchestrator: route -> tier check -> retrieve/fetch -> generate -> respond
"""

import asyncio
import logging
import queue

from shared.db import is_tracked, update_last_queried, promote_ticker
from question.models import AskRequest, QueryMode
from question.router import router
from question.retriever import retrieve
from question.live_fetcher import fetch_tier2_context
from question.tier2_cache import cache
from question.generator import generate_answer_stream

logger = logging.getLogger(__name__)


async def handle_ask_stream(req: AskRequest):
    """Streaming Q&A — returns an SSE generator."""
    route_result = router.route(req.query, req.tickers)
    tickers = route_result.tickers
    intent = route_result.intent
    mode = route_result.mode

    # Collect context first, then stream generation
    tier1_tickers = [t for t in tickers if is_tracked(t)]
    tier2_tickers = [t for t in tickers if t not in tier1_tickers]

    all_context_parts: list[str] = []
    data_freshness = ""

    if tier1_tickers:
        context, _, freshness = await retrieve(
            query=req.query, tickers=tier1_tickers, mode=mode, intent=intent,
        )
        all_context_parts.append(context)
        data_freshness = freshness

    for t in tier2_tickers:
        cached = cache.get(t, intent)
        if cached:
            all_context_parts.append(cached)
        else:
            live_context = await fetch_tier2_context(t)
            cache.put(t, intent, live_context)
            all_context_parts.append(live_context)

    full_context = "\n\n".join(all_context_parts) if all_context_parts else "(No data)"

    # Stream generation — run sync generator in thread to avoid blocking event loop.
    # kimi-k2.5 is a reasoning model: ~60s of reasoning chunks (no content)
    # followed by content chunks. Without threading, the sync for loop blocks
    # the event loop, causing SSE connections to die.
    q: queue.Queue = queue.Queue()
    SENTINEL = object()

    def _run_sync_gen():
        try:
            for token in generate_answer_stream(req.query, full_context, data_freshness):
                q.put(token)
        except Exception as e:
            logger.error(f"Generation thread error: {type(e).__name__}: {e}")
            q.put(e)
        finally:
            q.put(SENTINEL)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_sync_gen)

    while True:
        # Non-blocking poll with short sleep to keep event loop alive
        while q.empty():
            await asyncio.sleep(0.5)
            yield None  # signals keepalive to caller
        item = q.get_nowait()
        if item is SENTINEL:
            break
        if isinstance(item, Exception):
            logger.error(f"Generation thread raised: {item}")
            break
        yield item
