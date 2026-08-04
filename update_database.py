from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

import config  # forces USER_AGENT to be set immediately
from backend import scraper
from backend import vector_store


def update_database():
    """
    Description: Scrapes the target website, chunks the text, and uploads
        it to the vector database. This is the main entry point for the
        `update_database.py` script.
    Inputs: none. Reads global config.APP_ENV, config.CACHE_DIR.
    Outputs: returns a retriever object, or None if ingestion was skipped.
    Dependencies: calls backend.scraper.scrape_target_pages(),
        backend.vector_store.initialize_vector_db().
    Utilities: called by the `if __name__ == "__main__"` block below (i.e.
        this is the script's own entry point when run directly).
    """
    print("Scraping the target website...")
    documents = scraper.scrape_target_pages()

    if not documents:
        print(
            "Error: No scraped documents were returned, so vector ingestion was skipped."
        )
        return None

    print(
        f"Scraped {len(documents)} documents. Chunking and uploading to vector database..."
    )
    retriever = vector_store.initialize_vector_db(documents)

    if retriever is None:
        print("Error: Vector ingestion failed or was skipped.")
    else:
        print("Vector database updated successfully.")

    return retriever


if __name__ == "__main__":
    """
    Description: Main orchestration entry point for the `update_database.py`
        script. Scrapes the target website, chunks the text, and uploads it
        to the vector database.
    Inputs: none. No globals read directly (its dependencies do).
    Outputs: no return value.
    Dependencies: calls update_database().
    Utilities: called by the `if __name__ == "__main__"` block below (i.e. this is the script's own entry point when run directly).
    """
    update_database()
