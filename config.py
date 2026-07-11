import os

APP_ENV = os.getenv("APP_ENV", "local").strip().lower()
IS_LOCAL = APP_ENV == "local"
IS_PRODUCTION = APP_ENV == "production"

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chatbot/1.0"
USER_AGENT = os.getenv("USER_AGENT", DEFAULT_USER_AGENT)
os.environ.setdefault("USER_AGENT", USER_AGENT)

# Target links to scrape
TARGET_URLS = [
    "https://iitpkd.ac.in/",
    "https://iitpkd.ac.in/academics",
    "https://iitpkd.ac.in/admission",
    "https://iitpkd.ac.in/tenders",
]

# Vector store settings
CACHE_DIR = "./chroma_db" if IS_LOCAL else None
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "iit-pkd-index")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if IS_PRODUCTION and not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY must be set when APP_ENV=production")

# Text Processing Parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
