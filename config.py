import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Supabase REST API
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Moonshot (Kimi) API
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")
MOONSHOT_MODEL = os.getenv("MOONSHOT_MODEL", "kimi-k2.5")

# Token budget: reserve ~4k for system + prompt framing, rest for context
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "380000"))

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
