"""
question/models.py
All API request / response / internal data structures.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryMode(str, Enum):
    """Query mode — Router output, determines which retrieval path to use."""
    SEMANTIC   = "semantic"
    STRUCTURED = "structured"
    HYBRID     = "hybrid"


class Intent(str, Enum):
    """User intent — identified by Router, affects prompt and data strategy."""
    NEWS_SUMMARY      = "news_summary"
    EARNINGS_ANALYSIS = "earnings_analysis"
    PRICE_QUERY       = "price_query"
    TRADE_SUGGESTION  = "trade_suggestion"
    COMPARISON        = "comparison"
    GENERAL_ANALYSIS  = "general_analysis"


class RouterResult(BaseModel):
    tickers: List[str] = Field(default_factory=list, description="Extracted ticker symbols")
    intent: Intent = Field(default=Intent.GENERAL_ANALYSIS)
    mode: QueryMode = Field(default=QueryMode.HYBRID)


class Source(BaseModel):
    """Tracks where a piece of retrieved context came from (internal use)."""
    doc_type: str
    ticker: str
    date: Optional[str] = None
    title: Optional[str] = None
    similarity: Optional[float] = None


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    user_id: str = Field(default="anonymous")
    tickers: List[str] = Field(default_factory=list, description="Explicitly specified tickers")
