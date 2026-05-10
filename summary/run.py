"""
summary/run.py
Main entry point for the Summarization service.

Usage:
    python -m summary.run --ticker AAPL
    python -m summary.run --ticker AAPL,MSFT,NVDA
    python -m summary.run --all
    python -m summary.run --ticker AAPL --dry-run
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from config import OUTPUT_DIR, MOONSHOT_MODEL
from summary.fetcher import fetch_context
from summary.summarizer import generate_summary
from summary.prompts import PROMPT_VERSION
from summary.cache import compute_input_hash, get_cached, put_cached
from shared.db import get_tracked_tickers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")


def save_report(ticker: str, content: str) -> Path:
    today = date.today().isoformat()
    path = OUTPUT_DIR / f"{ticker}_{today}.md"
    path.write_text(content, encoding="utf-8")
    return path


def process_ticker(ticker: str, dry_run: bool = False) -> bool:
    """Fetch data, generate summary, and save report. Returns True on success.

    Caching: keyed on input_hash = SHA256(sorted source doc IDs + model +
    prompt_version). Identical inputs reuse the cached summary instead of
    re-calling the LLM.
    """
    try:
        ctx = fetch_context(ticker)

        if ctx.total_chars == 0:
            logger.warning(f"No data found for {ticker}, skipping.")
            return False

        input_hash = compute_input_hash(ctx, MOONSHOT_MODEL, PROMPT_VERSION)

        if dry_run:
            logger.info(
                f"[dry-run] {ticker}: {ctx.doc_counts}, {ctx.total_chars:,} chars, "
                f"input_hash={input_hash[:12]}..."
            )
            return True

        # 1) Try cache
        cached = get_cached(ticker, input_hash)
        if cached is not None:
            logger.info(
                f"Cache HIT for {ticker} (hash={input_hash[:12]}..., "
                f"originally generated {cached.summary_date})"
            )
            header = f"# {ticker} Investment Analysis Report\n\n"
            header += f"> Generated on {cached.summary_date.isoformat()} (cached)\n"
            header += f"> Data: {ctx.doc_counts}\n\n"
            path = save_report(ticker, header + cached.content)
            logger.info(f"Report saved from cache: {path}")
            return True

        # 2) Miss — call the LLM
        logger.info(f"Cache MISS for {ticker} (hash={input_hash[:12]}...) — calling LLM")
        report = generate_summary(ctx)

        # 3) Write back to cache (best-effort, won't raise)
        put_cached(
            ticker=ticker,
            input_hash=input_hash,
            content=report,
            model=MOONSHOT_MODEL,
            prompt_version=PROMPT_VERSION,
            source_doc_ids=ctx.source_doc_ids,
        )

        header = f"# {ticker} Investment Analysis Report\n\n"
        header += f"> Generated on {date.today().isoformat()}\n"
        header += f"> Data: {ctx.doc_counts}\n\n"

        path = save_report(ticker, header + report)
        logger.info(f"Report saved: {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {ticker}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate investment analysis reports via Kimi (Moonshot)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", type=str, help="Comma-separated ticker symbols (e.g. AAPL,MSFT)")
    group.add_argument("--all", action="store_true", help="Process all active tickers from Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Fetch data only, skip Moonshot API call")
    args = parser.parse_args()

    if args.all:
        rows = get_tracked_tickers(active_only=True)
        tickers = [r["ticker"] for r in rows]
        logger.info(f"Processing all {len(tickers)} active tickers: {tickers}")
    else:
        tickers = [t.strip().upper() for t in args.ticker.split(",")]

    results = {"success": [], "failed": [], "skipped": []}

    for ticker in tickers:
        ok = process_ticker(ticker, dry_run=args.dry_run)
        if ok:
            results["success"].append(ticker)
        else:
            results["failed"].append(ticker)

    logger.info("=" * 50)
    s, f = results["success"], results["failed"]
    logger.info(f"Done. Success: {s}, Failed: {f}")

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
