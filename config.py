"""
Description: Central settings file for the whole project. Every value here
    comes from an environment variable (set in .env locally, or in the
    Render dashboard in production), with a sensible default for local dev.
    This is the ONLY file that should call os.getenv() - every other module
    imports the finished values from here instead of reading the
    environment directly.
Inputs: none (this file has no functions - it runs once on import).
Outputs: none.
Dependencies: none.
Utilities: imported by every other module in the project (scraper.py,
    vector_store.py, chain_builder.py, cron_ingest.py, cli.py, wsgi.py, etc).
"""
import os

# ---------------------------------------------------------------------------
# App environment
# ---------------------------------------------------------------------------
APP_ENV = os.getenv("APP_ENV", "local").strip().lower()
IS_LOCAL = APP_ENV == "local"
IS_PRODUCTION = APP_ENV == "production"

# ---------------------------------------------------------------------------
# User agents (used both for the general HTTP client and the scraper)
# ---------------------------------------------------------------------------
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chatbot/1.0"
USER_AGENT = os.getenv("USER_AGENT", DEFAULT_USER_AGENT)
os.environ.setdefault("USER_AGENT", USER_AGENT)

DEFAULT_SCRAPER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
)

# ---------------------------------------------------------------------------
# Scraper settings
# ---------------------------------------------------------------------------
TARGET_URLS = [
    "https://iitpkd.ac.in/",
    "https://iitpkd.ac.in/academics",
    "https://iitpkd.ac.in/admission",
    "https://iitpkd.ac.in/tenders",
]

SCRAPER_BASE_URL = os.getenv("SCRAPER_BASE_URL", TARGET_URLS[0])
SCRAPER_DISCOVERY_MAX_PAGES = int(os.getenv("SCRAPER_DISCOVERY_MAX_PAGES", "25"))
SCRAPER_TARGET_MAX_PAGES = int(os.getenv("SCRAPER_TARGET_MAX_PAGES", "50"))
SCRAPER_REQUEST_TIMEOUT = int(os.getenv("SCRAPER_REQUEST_TIMEOUT", "10"))
SCRAPER_SKIP_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".zip", ".docx")
SCRAPER_HEADERS = {
    "User-Agent": os.getenv("SCRAPER_USER_AGENT", DEFAULT_SCRAPER_USER_AGENT)
}

# ---------------------------------------------------------------------------
# Text chunking settings
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Vector DB / embeddings settings
# ---------------------------------------------------------------------------
CACHE_DIR = "./chroma_db" if IS_LOCAL else None
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "iit-pkd-index")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "3"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google").strip().lower()
LOCAL_EMBEDDING_MODEL_NAME = os.getenv(
    "LOCAL_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_DB_RETRY_ATTEMPTS = int(os.getenv("VECTOR_DB_RETRY_ATTEMPTS", "4"))
VECTOR_DB_RETRY_SLEEP_BASE = float(os.getenv("VECTOR_DB_RETRY_SLEEP_BASE", "4"))
VECTOR_DB_RETRY_SLEEP_JITTER_MIN = float(
    os.getenv("VECTOR_DB_RETRY_SLEEP_JITTER_MIN", "1")
)
VECTOR_DB_RETRY_SLEEP_JITTER_MAX = float(
    os.getenv("VECTOR_DB_RETRY_SLEEP_JITTER_MAX", "3")
)

INGESTION_BATCH_SIZE = int(os.getenv("INGESTION_BATCH_SIZE", "10"))
INGESTION_RATE_LIMIT_COOLDOWN = int(os.getenv("INGESTION_RATE_LIMIT_COOLDOWN", "60"))
INGESTION_BATCH_PAUSE = int(os.getenv("INGESTION_BATCH_PAUSE", "8"))

if IS_PRODUCTION and not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY must be set when APP_ENV=production")

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen/qwen3.6-27b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# ---------------------------------------------------------------------------
# WhatsApp settings
# ---------------------------------------------------------------------------
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
WHATSAPP_GRAPH_API_VERSION = os.getenv("WHATSAPP_GRAPH_API_VERSION", "v18.0")
WHATSAPP_GRAPH_BASE_URL = os.getenv(
    "WHATSAPP_GRAPH_BASE_URL", "https://graph.facebook.com"
)
WHATSAPP_MESSAGE_TYPE = os.getenv("WHATSAPP_MESSAGE_TYPE", "text")