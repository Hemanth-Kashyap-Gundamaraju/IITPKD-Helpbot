import os
import sys
from dotenv import load_dotenv
import backend.scraper as scraper
import backend.vector_store as vector_store
import backend.chain_builder as chain_builder
import interface.cli as cli
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
    Description: Checks if a local Chroma database already exists on disk.
        If it does, we don't need to re-scrape and re-embed everything
        again - we can just load it and start chatting, saving our daily
        embedding quota.
    Inputs: none. Reads global config.CACHE_DIR.
    Outputs: returns True/False. No globals changed.
    Dependencies: none.
    Utilities: called by main().
    """
    return (
        config.CACHE_DIR is not None
        and os.path.exists(config.CACHE_DIR)
        and len(os.listdir(config.CACHE_DIR)) > 0
    )


def _user_requested_refresh():
    """
    Description: Checks if the user passed --refresh to force a fresh
        re-scrape and re-embed.
    Inputs: none. Reads global FORCE_REFRESH_FLAG, sys.argv.
    Outputs: returns True/False. No globals changed.
    Dependencies: none.
    Utilities: called by main().
    """
    return FORCE_REFRESH_FLAG in sys.argv


def _build_chain_from_fresh_data():
    """
    Description: Scrapes the website, embeds everything, and builds a new
        RAG chain. This is the "expensive" path - it burns embedding
        quota, so it's only taken when the cache is missing or the user
        explicitly asked for --refresh.
    Inputs: none. No globals read directly (its dependencies do).
    Outputs: returns a RAG chain (LangChain runnable). No globals changed.
    Dependencies: calls backend.scraper.scrape_target_pages(),
        backend.vector_store.initialize_vector_db(),
        backend.chain_builder.build_rag_chain().
    Utilities: called by main().
    """
    print("Building a fresh vector database (this uses embedding API calls)...")
    raw_docs = scraper.scrape_target_pages()
    retriever = vector_store.initialize_vector_db(raw_docs)
    return chain_builder.build_rag_chain(retriever)


def _build_chain_from_existing_data():
    """
    Description: Loads the already-existing local vector database without
        re-embedding anything. This is the "cheap" path taken on every
        normal run once the cache exists.
    Inputs: none. No globals read directly.
    Outputs: returns a RAG chain (LangChain runnable). No globals changed.
    Dependencies: calls backend.chain_builder.build_rag_chain() (passing
        retriever=None tells it to load the existing Chroma DB from disk).
    Utilities: called by main().
    """
    print("Found an existing local vector database - loading it (no API calls needed).")
    return chain_builder.build_rag_chain(retriever=None)


def main():
    """
    Description: Main orchestration entry point - decides whether to
        rebuild the vector database or reuse the existing one, then hands
        off to the CLI chat loop.
    Inputs: none. No globals read directly (its dependencies do).
    Outputs: no return value.
    Dependencies: calls _user_requested_refresh(),
        _local_database_already_exists(), _build_chain_from_fresh_data(),
        _build_chain_from_existing_data(), interface.cli.run_chat_loop().
    Utilities: called by the `if __name__ == "__main__"` block below (i.e.
        this is the script's own entry point when run directly).
    """
    if _user_requested_refresh() or not _local_database_already_exists():
        rag_chain = _build_chain_from_fresh_data()
    else:
        rag_chain = _build_chain_from_existing_data()

    cli.run_chat_loop(rag_chain)


if __name__ == "__main__":
    main()