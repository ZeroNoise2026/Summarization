"""
run.py
Main entry point for the Summarization service.

Usage:
    python run.py --ticker AAPL              # single ticker
    python run.py --ticker AAPL,MSFT,NVDA    # multiple tickers
    python run.py --all                      # all active tickers from Supabase
    python run.py --ticker AAPL --dry-run    # fetch data only, skip API call
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from config import OUTPUT_DIR
from fetcher import fetch_context
from summarizer import generate_summary
from store.db import get_tracked_tickers

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
    """Fetch data, generate summary, and save report. Returns True on success."""
    try:
        ctx = fetch_context(ticker)

        if ctx.total_chars == 0:
            logger.warning(f"No data found for {ticker}, skipping.")
            return False

        if dry_run:
            logger.info(f"[dry-run] {ticker}: {ctx.doc_counts}, {ctx.total_chars:,} chars total")
            return True

        report = generate_summary(ctx)

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
    logger.info(f"Done. Success: {results['success']}, Failed: {results['failed']}")

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
