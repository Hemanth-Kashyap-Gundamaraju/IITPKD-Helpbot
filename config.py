import os

# Target links to scrape
TARGET_URLS = [
    "https://iitpkd.ac.in/",
    "https://iitpkd.ac.in/academics",
    "https://iitpkd.ac.in/admission",
    "https://iitpkd.ac.in/tenders"
]

# Chroma DB Local Cache Location
CACHE_DIR = "./chroma_db"

# Text Processing Parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# User agent string to prevent web scraping blocks
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chatbot/1.0"
os.environ["USER_AGENT"] = USER_AGENT