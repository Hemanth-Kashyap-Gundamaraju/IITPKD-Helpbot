import os
import sys
from dotenv import load_dotenv
import backend.scraper as scraper
import backend.vector_store as vector_store
import backend.chain_builder as chain_builder
import cli as cli
import config

load_dotenv()

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# Pass "--refresh" as a command line flag when you actually want to
# re-scrape the website and re-embed everything from scratch.
# Example: python main.py --refresh
FORCE_REFRESH_FLAG = "--refresh"


def _local_database_already_exists():
    """
    Checks if a local Chroma database already exists on disk.
    If it does, we don't need to re-scrape and re-embed everything again -
    we can just load it and start chatting, saving our daily embedding quota.
    """
    return (
        config.CACHE_DIR is not None
        and os.path.exists(config.CACHE_DIR)
        and len(os.listdir(config.CACHE_DIR)) > 0
    )


def _user_requested_refresh():
    """Checks if the user passed --refresh to force a fresh re-scrape and re-embed."""
    return FORCE_REFRESH_FLAG in sys.argv


def _build_chain_from_fresh_data():
    """Scrapes the website, embeds everything, and builds a new RAG chain."""
    print("Building a fresh vector database (this uses embedding API calls)...")
    raw_docs = scraper.scrape_target_pages()
    retriever = vector_store.initialize_vector_db(raw_docs)
    return chain_builder.build_rag_chain(retriever)


def _build_chain_from_existing_data():
    """Loads the already-existing local vector database without re-embedding anything."""
    print("Found an existing local vector database - loading it (no API calls needed).")
    # Passing retriever=None here makes chain_builder load the existing
    # Chroma database from disk instead of creating a new one.
    return chain_builder.build_rag_chain(retriever=None)


def main():
    """Main orchestration pipeline workflow entrypoint."""
    if _user_requested_refresh() or not _local_database_already_exists():
        rag_chain = _build_chain_from_fresh_data()
    else:
        rag_chain = _build_chain_from_existing_data()

    cli.run_chat_loop(rag_chain)


if __name__ == "__main__":
    main()