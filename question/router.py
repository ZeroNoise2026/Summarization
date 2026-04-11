"""
question/router.py
Two-layer router: L1 keyword matching (free) -> L2 KIMI fallback (when L1 misses)

L1: Regex ticker extraction + keyword intent matching — zero cost, <1ms
L2: KIMI classification — ~100 tokens, ~500ms, only when L1 score=0 or no tickers found
"""

import json
import re
import logging
from typing import Optional

from question.models import Intent, QueryMode, RouterResult

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# Company aliases -> ticker mapping
# ═══════════════════════════════════════════

COMPANY_ALIASES: dict[str, str] = {
    "Apple": "AAPL", "Amazon": "AMZN", "Google": "GOOGL",
    "Alphabet": "GOOGL", "Meta": "META", "Facebook": "META",
    "Microsoft": "MSFT", "Nvidia": "NVDA", "Tesla": "TSLA",
    "S&P": "SPY", "Nasdaq": "QQQ",
    # Commonly referenced by description, not ticker
    "ON Semiconductor": "ON", "ServiceNow": "NOW", "Allstate": "ALL",
    "Ford": "F", "Visa": "V", "AT&T": "T", "US Steel": "X",
    "Alibaba": "BABA", "JPMorgan": "JPM", "Goldman": "GS",
    "iPhone maker": "AAPL", "search giant": "GOOGL",
    # Cloud providers — implicit ticker mapping
    "AWS": "AMZN", "Azure": "MSFT", "GCP": "GOOGL",
}

# Phrases that map to multiple tickers (one-to-many)
PHRASE_TO_TICKERS: dict[str, list[str]] = {
    "cloud provider": ["AMZN", "MSFT", "GOOGL"],
    "cloud computing": ["AMZN", "MSFT", "GOOGL"],
    "big tech": ["AAPL", "AMZN", "GOOGL", "META", "MSFT"],
    "magnificent seven": ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"],
    "mag 7": ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"],
    "faang": ["META", "AAPL", "AMZN", "NVDA", "GOOGL"],
}

# ═══════════════════════════════════════════
# Intent keyword table
# ═══════════════════════════════════════════

INTENT_KEYWORDS: dict[Intent, list[str]] = {
    Intent.NEWS_SUMMARY: [
        "news", "headline", "what happened",
        "recent news", "latest news", "announcement", "development",
    ],
    Intent.EARNINGS_ANALYSIS: [
        "earnings", "revenue", "income", "profit", "quarterly",
        "EPS", "net income", "gross margin", "financial",
        "latest earnings", "earnings report",
    ],
    Intent.PRICE_QUERY: [
        "price", "quote", "market cap", "valuation",
        "how much", "stock price", "PE", "P/E ratio",
    ],
    Intent.TRADE_SUGGESTION: [
        "buy", "sell", "hold", "recommend", "should I",
        "worth buying", "investment", "trade", "position",
    ],
    Intent.COMPARISON: [
        "compare", "versus", "vs", "better", "difference",
        "which one", "head to head",
    ],
    Intent.GENERAL_ANALYSIS: [
        "overview", "general", "how is", "how are",
        "tell me about", "what about", "outlook",
        "doing financially", "update on", "analysis of",
    ],
}

# Tie-breaking priority: higher = wins when scores are equal
# COMPARISON and TRADE are more specific intents — should win ties
INTENT_PRIORITY: dict[Intent, int] = {
    Intent.NEWS_SUMMARY:      1,
    Intent.EARNINGS_ANALYSIS: 2,
    Intent.PRICE_QUERY:       2,
    Intent.TRADE_SUGGESTION:  3,
    Intent.COMPARISON:        4,
    Intent.GENERAL_ANALYSIS:  0,
}

# Intent -> retrieval mode mapping
INTENT_TO_MODE: dict[Intent, QueryMode] = {
    Intent.NEWS_SUMMARY:      QueryMode.SEMANTIC,     # news -> semantic search
    Intent.EARNINGS_ANALYSIS: QueryMode.HYBRID,        # earnings -> semantic + structured
    Intent.PRICE_QUERY:       QueryMode.STRUCTURED,    # price -> pure structured
    Intent.TRADE_SUGGESTION:  QueryMode.HYBRID,        # suggestion -> need all data
    Intent.COMPARISON:        QueryMode.HYBRID,        # comparison -> need all data
    Intent.GENERAL_ANALYSIS:  QueryMode.HYBRID,        # fallback -> a bit of everything
}

# Common English words to exclude from ticker extraction
# Known single-letter tickers — DO NOT put these in COMMON_WORDS
# F=Ford, V=Visa, C=Citigroup, X=US Steel, T=AT&T
SINGLE_LETTER_TICKERS = {"F", "V", "C", "X", "T"}

# Ambiguous tickers that are also very common English words.
# These are ONLY blocked from regex extraction; L2 KIMI can still return them.
# Real tickers removed (query them via L2): ALL, NOW, HAS, IPO, etc.
COMMON_WORDS = {
    # Single letters (excluding real tickers in SINGLE_LETTER_TICKERS)
    "I", "A", "B", "D", "E", "G", "H", "K",
    "M", "N", "O", "P", "Q", "R", "S", "U", "W", "Y",
    # 2-letter common words (ON=ON Semi, BE=Bloom Energy — too common as English)
    "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO",
    "HE", "IF", "IN", "IS", "IT", "ME", "MY", "NO",
    "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    # 3-letter common words (removed real tickers: ALL, NOW, HAS, ANY, CAN, NEW, OUT, OWN, OLD)
    "THE", "AND", "FOR", "NOT", "ARE", "BUT", "WAS",
    "HER", "HIS", "HOW", "ITS", "MAY", "OUR", "SAY",
    "SHE", "TOO", "USE", "WHO", "WHY", "YET", "GOT",
    "LET", "PUT", "RUN", "SET", "TRY", "WAY", "VS",
    # 4+ letter common words
    "THIS", "THAT", "WITH", "FROM", "WILL", "WHAT",
    "BEEN", "HAVE", "JUST", "MORE", "ALSO", "THAN",
    # Finance terms (not tickers)
    "BUY", "SELL", "HOLD", "EPS", "CEO", "CFO",
    "ETF", "GDP", "API", "USA", "IPO", "FAANG",
}


class QueryRouter:
    """L1 keyword router, with L2 KIMI fallback when keywords miss."""

    # ── L2 KIMI classification prompt (kept minimal for cost) ──
    _CLASSIFY_SYSTEM = (
        "You are a financial query classifier. "
        "Given a user question, return ONLY a JSON object with two fields:\n"
        '  "intent": one of ["news_summary","earnings_analysis","price_query","trade_suggestion","comparison","general_analysis"]\n'
        '  "tickers": a list of US stock ticker symbols mentioned or implied (e.g. ["AAPL","MSFT"]). '
        "Use standard ticker symbols. Return [] if no specific company is mentioned.\n"
        "Return ONLY valid JSON, no explanation."
    )

    _CLASSIFY_EXAMPLES = [
        {"role": "user", "content": "How is the iPhone maker doing financially?"},
        {"role": "assistant", "content": '{"intent":"general_analysis","tickers":["AAPL"]}'},
        {"role": "user", "content": "Which cloud provider has better margins, AWS or Azure?"},
        {"role": "assistant", "content": '{"intent":"comparison","tickers":["AMZN","MSFT"]}'},
        {"role": "user", "content": "What happened in the market today?"},
        {"role": "assistant", "content": '{"intent":"news_summary","tickers":[]}'},
    ]

    def route(self, query: str, explicit_tickers: Optional[list[str]] = None) -> RouterResult:
        tickers = self._extract_tickers(query)
        if explicit_tickers:
            tickers = list(set(tickers + [t.upper() for t in explicit_tickers]))

        intent, score = self._detect_intent(query)

        # L2 KIMI fallback — when keyword matching completely misses
        # Triggers when: (1) intent score=0, OR (2) no tickers found and no explicit tickers
        needs_l2 = (score == 0) or (len(tickers) == 0 and not explicit_tickers)

        if needs_l2:
            logger.info(
                f"L1 incomplete (score={score}, tickers={tickers}) "
                f"for: \"{query[:60]}\" -> calling L2 KIMI classify"
            )
            l2_intent, l2_tickers = self._kimi_classify(query)

            if l2_intent is not None and score == 0:
                intent = l2_intent
            # Only use L2 tickers if L1 found none — never let L2 inject
            # extra tickers when L1 already identified the relevant ones.
            if l2_tickers and len(tickers) == 0:
                tickers = l2_tickers
            elif l2_tickers and len(tickers) > 0:
                logger.info(f"L2 returned extra tickers {l2_tickers}, keeping L1 tickers {tickers}")

        mode = INTENT_TO_MODE.get(intent, QueryMode.HYBRID)
        return RouterResult(tickers=sorted(tickers), intent=intent, mode=mode)

    def _kimi_classify(self, query: str) -> tuple[Optional[Intent], list[str]]:
        """L2 fallback: ask KIMI to classify intent + extract tickers.

        Returns (intent_or_None, tickers_list).
        On any failure, returns (None, []) — the caller keeps L1 defaults.
        Cost: ~150 tokens total (system+examples+query+response), ~500ms.
        """
        from shared.llm import chat, has_api_key
        from config import KIMI_MODEL_CLASSIFY

        if not has_api_key():
            logger.warning("MOONSHOT_API_KEY not set — L2 KIMI fallback disabled")
            return None, []

        messages = [
            {"role": "system", "content": self._CLASSIFY_SYSTEM},
            *self._CLASSIFY_EXAMPLES,
            {"role": "user", "content": query},
        ]

        try:
            raw, usage = chat(
                messages=messages,
                model=KIMI_MODEL_CLASSIFY,
                temperature=0,       # deterministic classification
                max_tokens=150,      # moonshot-v1-8k: concise JSON, no reasoning overhead
                max_retries=1,       # don't retry — keep router fast
                timeout=8.0,         # moonshot-v1-8k is faster (~1s), 8s generous timeout
            )
            raw = raw.strip()
            logger.info(
                f"L2 KIMI classify: tokens={usage['total_tokens']}, raw={raw[:120]}"
            )
            return self._parse_classify_response(raw)

        except Exception as e:
            logger.warning(f"L2 KIMI classify failed, falling back to L1 defaults: {e}")
            return None, []

    def _parse_classify_response(self, raw: str) -> tuple[Optional[Intent], list[str]]:
        """Parse KIMI's JSON response into (Intent, tickers).

        Handles: plain JSON, markdown-fenced JSON, partial garbage.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove ```json ... ``` wrapper
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to repair truncated JSON (e.g. '{"intent":"price_query","tick')
            data = self._repair_partial_json(cleaned)
            if data is None:
                logger.warning(f"L2 response not valid JSON (unrepairable): {raw[:120]}")
                return None, []
            logger.info(f"L2 partial JSON repaired: {data}")

        # Parse intent
        intent = None
        raw_intent = data.get("intent", "")
        try:
            intent = Intent(raw_intent)
        except ValueError:
            logger.warning(f"L2 returned unknown intent: {raw_intent}")

        # Parse tickers
        raw_tickers = data.get("tickers", [])
        tickers = []
        if isinstance(raw_tickers, list):
            for t in raw_tickers:
                if isinstance(t, str) and 1 <= len(t) <= 5 and t.isalpha():
                    tickers.append(t.upper())

        return intent, tickers

    @staticmethod
    def _repair_partial_json(raw: str) -> Optional[dict]:
        """Try to extract intent and tickers from a truncated JSON string.
        
        Handles cases like: '{"intent":"price_query","tick' (cut off at max_tokens)
        """
        result = {}
        # Extract intent
        intent_match = re.search(r'"intent"\s*:\s*"([a-z_]+)"', raw)
        if intent_match:
            result["intent"] = intent_match.group(1)
        # Extract tickers array (may be partial)
        tickers_match = re.search(r'"tickers"\s*:\s*\[([^\]]*)\]?', raw)
        if tickers_match:
            ticker_values = re.findall(r'"([A-Z]{1,5})"', tickers_match.group(1))
            result["tickers"] = ticker_values
        return result if result else None

    def _extract_tickers(self, query: str) -> list[str]:
        """Extract ticker symbols from query text.
        
        Sources:
          1. $-prefixed symbols (e.g. $F, $AAPL) — always accepted
          2. Regex: uppercase 1-5 letter words, filtered by COMMON_WORDS
             - Single-letter matches only accepted if in SINGLE_LETTER_TICKERS
               AND query is short (likely just the ticker)
          3. Company alias mapping (e.g. "Apple" -> AAPL)
        """
        tickers: set[str] = set()

        # Source 1: $-prefixed tickers (always trusted)
        dollar_matches = re.findall(r"\$([A-Z]{1,5})\b", query)
        for match in dollar_matches:
            tickers.add(match)

        # Source 2: regex match uppercase letter groups
        regex_matches = re.findall(r"\b([A-Z]{1,5})\b", query)
        for match in regex_matches:
            if match in COMMON_WORDS:
                continue
            # Single-letter: only accept known tickers in reasonably short queries
            if len(match) == 1:
                if match in SINGLE_LETTER_TICKERS and len(query.split()) <= 12:
                    tickers.add(match)
                continue
            tickers.add(match)

        # Source 3: company alias matching (case-insensitive)
        query_lower = query.lower()
        for alias, ticker in COMPANY_ALIASES.items():
            if alias.lower() in query_lower:
                tickers.add(ticker)

        # Source 4: phrase-to-tickers (one phrase -> multiple tickers)
        for phrase, phrase_tickers in PHRASE_TO_TICKERS.items():
            if phrase in query_lower:
                tickers.update(phrase_tickers)

        return list(tickers)

    def _detect_intent(self, query: str) -> tuple[Intent, int]:
        """Detect user intent via keyword matching.
        
        Returns (intent, score). score=0 means L1 completely missed.
        Uses priority tie-breaking: when two intents have the same score,
        the more specific intent (higher priority) wins.
        
        Short keywords (<=3 chars, e.g. "PE", "EPS", "vs") use word-boundary
        matching to avoid false substring hits (e.g. "happened" matching "PE").
        """
        query_lower = query.lower()
        best_intent = Intent.GENERAL_ANALYSIS
        best_score = 0
        best_priority = 0
        for intent, keywords in INTENT_KEYWORDS.items():
            score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                if len(kw) <= 3:
                    # Short keywords: require word boundary to avoid false substring matches
                    if re.search(r'\b' + re.escape(kw_lower) + r'\b', query_lower):
                        score += 1
                else:
                    if kw_lower in query_lower:
                        score += 1
            priority = INTENT_PRIORITY.get(intent, 0)
            if score > best_score or (score == best_score and score > 0 and priority > best_priority):
                best_score = score
                best_intent = intent
                best_priority = priority
        return best_intent, best_score


# Module-level singleton
router = QueryRouter()
