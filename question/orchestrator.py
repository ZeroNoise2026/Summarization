"""
question/orchestrator.py
Main orchestrator: route -> tier check -> retrieve/fetch -> generate -> respond

Fast paths (skip LLM, <2s):
  - price_query: direct DB lookup + live quote → formatted response
"""

import asyncio
import logging
import queue
import time

from shared.db import is_tracked, update_last_queried, promote_ticker
from shared.db import get_earnings, get_price_snapshots
from question.models import AskRequest, Intent, QueryMode
from question.router import router
from question.retriever import retrieve
from question.live_fetcher import fetch_tier2_context, fetch_live_quote_raw
from question.tier2_cache import cache
from question.generator import generate_answer_stream

logger = logging.getLogger(__name__)


def _fmt_money(val) -> str:
    """Format a number as human-readable money (e.g. $24.9B, $840M, $1.2K)."""
    if val is None:
        return "N/A"
    v = abs(float(val))
    sign = "-" if float(val) < 0 else ""
    if v >= 1e12:
        return f"{sign}${v/1e12:.2f}T"
    if v >= 1e9:
        return f"{sign}${v/1e9:.1f}B"
    if v >= 1e6:
        return f"{sign}${v/1e6:.0f}M"
    if v >= 1e3:
        return f"{sign}${v/1e3:.0f}K"
    return f"{sign}${v:,.0f}"


# ═══════════════════════════════════════════
# Fast path: price_query — skip LLM, <2s
# ═══════════════════════════════════════════

async def _fast_price_response(tickers: list[str]) -> str:
    """Build a formatted price response directly from DB + live API.

    Returns a human-readable Markdown string — no LLM needed.
    P/E is calculated from price / TTM EPS when not in the live quote.
    """
    parts: list[str] = []

    for ticker in tickers:
        section: list[str] = [f"## {ticker}"]

        # 1. Fetch live quote + earnings in parallel-ish
        live = await fetch_live_quote_raw(ticker)
        earnings = get_earnings(ticker, limit=4)

        # 2. Calculate TTM P/E from earnings if live quote doesn't include it
        pe = live.get("pe") if live else None
        if pe is None and live and live.get("price") and earnings:
            eps_vals = [e["eps"] for e in earnings if e.get("eps") is not None]
            if eps_vals:
                ttm_eps = sum(eps_vals[:4])
                if ttm_eps > 0:
                    pe = live["price"] / ttm_eps

        # 3. Build price section
        if live:
            price = live.get("price")
            change = live.get("changePercentage")
            mc = live.get("marketCap")
            day_high = live.get("dayHigh")
            day_low = live.get("dayLow")
            year_high = live.get("yearHigh")
            year_low = live.get("yearLow")
            volume = live.get("volume")
            avg50 = live.get("priceAvg50")
            avg200 = live.get("priceAvg200")
            name = live.get("name", ticker)

            # NOTE: Each bold line ends with two trailing spaces ("  ")
            # so that Markdown renders them as <br> line breaks.
            # Without this, single \n between lines is collapsed to a space.
            section.append(f"**{name}**  ")
            section.append(f"**Price:** ${price:,.2f}  " if price else "**Price:** N/A  ")
            if change is not None:
                arrow = "🟢" if change >= 0 else "🔴"
                section.append(f"**Change:** {arrow} {change:+.2f}%  ")
            if day_high and day_low:
                section.append(f"**Day Range:** ${day_low:,.2f} – ${day_high:,.2f}  ")
            if year_high and year_low:
                section.append(f"**52W Range:** ${year_low:,.2f} – ${year_high:,.2f}  ")
            if pe:
                section.append(f"**P/E Ratio:** {pe:.1f}  ")
            if mc:
                if mc >= 1e12:
                    section.append(f"**Market Cap:** ${mc/1e12:.2f}T  ")
                elif mc >= 1e9:
                    section.append(f"**Market Cap:** ${mc/1e9:.2f}B  ")
                else:
                    section.append(f"**Market Cap:** ${mc/1e6:.0f}M  ")
            if volume:
                section.append(f"**Volume:** {volume:,.0f}  ")
            if avg50:
                section.append(f"**50D Avg:** ${avg50:,.2f}  ")
            if avg200:
                section.append(f"**200D Avg:** ${avg200:,.2f}  ")
        else:
            # Fallback to DB snapshot
            snapshots = get_price_snapshots(ticker, limit=1)
            if snapshots:
                s = snapshots[0]
                section.append(f"**Price:** ${s['close_price']:,.2f} (as of {s['date']})  ")
                db_pe = s.get("pe_ratio") or pe  # use calculated P/E as fallback
                if db_pe:
                    section.append(f"**P/E Ratio:** {db_pe:.1f}  ")
                if s.get("market_cap"):
                    mc = s["market_cap"]
                    if mc >= 1e12:
                        section.append(f"**Market Cap:** ${mc/1e12:.2f}T  ")
                    else:
                        section.append(f"**Market Cap:** ${mc/1e9:.2f}B  ")
            else:
                section.append("*No price data available.*")

        # 4. Recent earnings from DB (already fetched above)
        if earnings:
            section.append("")
            section.append("### Recent Earnings")
            section.append("| Quarter | EPS | Revenue | Net Income |")
            section.append("|---------|-----|---------|------------|")
            for e in earnings:
                eps = f"${e['eps']:.2f}" if e.get("eps") is not None else "N/A"
                rev = _fmt_money(e.get("revenue")) if e.get("revenue") and e["revenue"] != 0 else "N/A"
                ni = _fmt_money(e.get("net_income")) if e.get("net_income") else "N/A"
                section.append(f"| {e['quarter']} | {eps} | {rev} | {ni} |")

        parts.append("\n".join(section))

    return "\n\n".join(parts)


async def handle_ask_stream(req: AskRequest):
    """Streaming Q&A — returns an SSE generator.

    Fast path: price_query intent → direct DB/API lookup, no LLM (~1-2s)
    Normal path: all other intents → retrieve context → LLM generation (~50-90s)
    """
    route_result = router.route(req.query, req.tickers)
    tickers = route_result.tickers
    intent = route_result.intent
    mode = route_result.mode
    yield ("status", "🔍 Analyzing your question...")
    # ── Default market tickers for ticker-less market queries ───────
    # When user asks "What happened in the market today?" with no tickers,
    # inject SPY/QQQ so we have some context to work with.
    if not tickers and intent in (Intent.NEWS_SUMMARY, Intent.GENERAL_ANALYSIS):
        logger.info("No tickers found for market query — injecting SPY, QQQ as defaults")
        tickers = ["QQQ", "SPY"]

    # ── Fast path: price_query ──────────────────────────────────────
    if intent == Intent.PRICE_QUERY and tickers:
        logger.info(f"⚡ Fast path: price_query for {tickers}")
        t0 = time.time()
        response = await _fast_price_response(tickers)
        latency = int((time.time() - t0) * 1000)
        logger.info(f"⚡ Fast path completed in {latency}ms")

        # Yield the formatted response as tokens (simulating streaming)
        # Send in chunks so the UI renders progressively
        for line in response.split("\n"):
            yield ("token", line + "\n")
        return

    # ── Normal path: retrieve + LLM ────────────────────────────────
    tier1_tickers = [t for t in tickers if is_tracked(t)]
    tier2_tickers = [t for t in tickers if t not in tier1_tickers]
    yield ("status", f"📡 Searching data for {', '.join(tickers)}...")
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

    yield ("status", "🤖 Generating answer...")

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
