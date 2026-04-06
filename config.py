"""
config.py
All environment variables for the Summarization repo — single source of truth.

Sections:
  1. Supabase (shared)
  2. Moonshot / KIMI (shared)
  3. Summary-service settings
  4. Question-service settings
  5. Upstream service URLs
  6. Vector search parameters
  7. Tier 2 cache parameters
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════
# 1. Supabase (shared by summary + question)
# ═══════════════════════════════════════════
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ═══════════════════════════════════════════
# 2. Moonshot / KIMI credentials (shared)
# ═══════════════════════════════════════════
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")

# ═══════════════════════════════════════════
# 3. Summary-service settings
# ═══════════════════════════════════════════
MOONSHOT_MODEL = os.getenv("MOONSHOT_MODEL", "kimi-k2.5")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "380000"))
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════
# 4. Question-service LLM settings
# ═══════════════════════════════════════════
KIMI_MODEL_QA: str = os.getenv("KIMI_MODEL_QA", "kimi-k2.5")
KIMI_MODEL_CLASSIFY: str = os.getenv("KIMI_MODEL_CLASSIFY", "moonshot-v1-8k")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ═══════════════════════════════════════════
# 5. Upstream service URLs
# ═══════════════════════════════════════════
EMBEDDING_SERVICE_URL: str = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8002")
DATA_PIPELINE_URL: str = os.getenv("DATA_PIPELINE_URL", "http://localhost:8001")

# ═══════════════════════════════════════════
# 6. Vector search parameters
# ═══════════════════════════════════════════
SEMANTIC_TOP_K: int = int(os.getenv("SEMANTIC_TOP_K", "8"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# ═══════════════════════════════════════════
# 7. Tier 2 cache parameters
# ═══════════════════════════════════════════
TIER2_CACHE_MAX_SIZE: int = int(os.getenv("TIER2_CACHE_MAX_SIZE", "200"))
TIER2_CACHE_TTL_PRICE: int = int(os.getenv("TIER2_CACHE_TTL_PRICE", "300"))
TIER2_CACHE_TTL_NEWS: int = int(os.getenv("TIER2_CACHE_TTL_NEWS", "3600"))
TIER2_CACHE_TTL_EARNINGS: int = int(os.getenv("TIER2_CACHE_TTL_EARNINGS", "86400"))
TIER2_CACHE_TTL_DEFAULT: int = int(os.getenv("TIER2_CACHE_TTL_DEFAULT", "1800"))

# ── Hot ticker auto-promotion threshold ──
HOT_PROMOTION_THRESHOLD: int = int(os.getenv("HOT_PROMOTION_THRESHOLD", "10"))

