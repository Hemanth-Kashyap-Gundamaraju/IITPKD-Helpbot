# =====================================================================
# CRITICAL: This environment patch MUST run before any other imports!
# =====================================================================
import os
from dotenv import load_dotenv

load_dotenv()

import config

os.environ["USER_AGENT"] = config.SCRAPER_HEADERS["User-Agent"]
# =====================================================================

import time
from langchain_pinecone import PineconeVectorStore
from backend import scraper
from backend.embeddings_utils import get_embeddings
from backend.chunking_utils import chunk_documents
from backend.batching_utils import split_into_batches

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# How many chunks we upload to Pinecone in one go (kept small to avoid
# tripping Google's free-tier embedding rate limit).
UPLOAD_BATCH_SIZE = config.INGESTION_BATCH_SIZE

# How long we pause between two normal, successful batch uploads.
BATCH_PAUSE_SECONDS = config.INGESTION_BATCH_PAUSE

# How long we wait after Google tells us "you're over quota, slow down".
RATE_LIMIT_COOLDOWN_SECONDS = config.INGESTION_RATE_LIMIT_COOLDOWN


def _is_quota_error(error):
    """
    Description: Checks if an error is a '429 RESOURCE_EXHAUSTED' quota error.
    Inputs: error (the exception object). No globals read.
    Outputs: returns True/False. No globals changed.
    Dependencies: none.
    Utilities: called by PineconeBatchUploader.upload_batch_with_retry().
    """
    error_text = str(error)
    return "429" in error_text or "RESOURCE_EXHAUSTED" in error_text


def _scrape_raw_documents():
    """
    Description: Scrapes the college website and returns the raw documents.
        The only function in this file that touches the network for
        scraping, so the rest of the pipeline can be tested without
        actually hitting the college website.
    Inputs: none.
    Outputs: returns a list of LangChain Document objects. No globals changed.
    Dependencies: calls scraper.scrape_target_pages().
    Utilities: called by run_ingestion().
    """
    print("Starting Production Data Ingestion Pipeline...")
    return scraper.scrape_target_pages()


def _has_usable_content(raw_docs):
    """
    Description: Checks whether any of the scraped pages actually contain text.
        Pure check, no network calls, so it's easy to test with fake documents.
    Inputs: raw_docs (list of Document objects).
    Outputs: returns True/False. No globals changed.
    Dependencies: none.
    Utilities: called by run_ingestion().
    """
    if not raw_docs:
        return False
    return any(len(doc.page_content.strip()) > 0 for doc in raw_docs)


class PineconeBatchUploader:
    """
    Description: Handles uploading chunk batches to Pinecone. Groups together
        the embeddings model, the target index name, and the growing
        vector_store connection - the same three things every upload step
        needs - as attributes on one object instead of passing them
        separately into every function.
    Inputs (constructor): embeddings (the embeddings model to use), index_name
        (the Pinecone index to upload into).
    Utilities: used by run_ingestion().
    """

    def __init__(self, embeddings, index_name):
        self.embeddings = embeddings
        self.index_name = index_name
        self.vector_store = None  # set once the first batch creates the index

    def upload_batch(self, batch):
        """
        Description: Uploads one batch of chunks to Pinecone. Creates the
            index on the first call, adds to it on every call after that.
            This is the only method that actually talks to Pinecone, so
            it's the only piece that needs a real (or mocked) network
            connection to test.
        Inputs: batch (list of Document chunks). Reads self.embeddings,
            self.index_name, self.vector_store.
        Outputs: no return value. Sets self.vector_store (on the first call).
        Dependencies: uses langchain_pinecone.PineconeVectorStore.
        Utilities: called by upload_batch_with_retry().
        """
        if self.vector_store is None:
            self.vector_store = PineconeVectorStore.from_documents(
                documents=batch, embedding=self.embeddings, index_name=self.index_name
            )
        else:
            self.vector_store.add_documents(documents=batch)

    def upload_batch_with_retry(self, batch, batch_number, total_batches):
        """
        Description: Tries to upload one batch, retrying once after a
            cooldown if we hit Google's rate limit. Any other error is
            re-raised as-is.
        Inputs: batch, batch_number, total_batches. Reads global
            RATE_LIMIT_COOLDOWN_SECONDS.
        Outputs: no return value (self.vector_store is updated via upload_batch).
        Dependencies: calls upload_batch(), _is_quota_error().
        Utilities: called by upload_all_batches().
        """
        try:
            self.upload_batch(batch)
        except Exception as e:
            if not _is_quota_error(e):
                raise

            print(
                f"   [Rate Limit] Batch {batch_number}/{total_batches} hit the free-tier "
                f"quota. Cooling down for {RATE_LIMIT_COOLDOWN_SECONDS}s..."
            )
            time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
            self.upload_batch(batch)

    def upload_all_batches(self, chunks):
        """
        Description: Uploads every chunk batch to Pinecone, pausing between
            batches to stay under the rate limit. Orchestrates the other
            methods on this class rather than doing any single thing itself.
        Inputs: chunks (full list of chunked Documents). Reads global
            UPLOAD_BATCH_SIZE, BATCH_PAUSE_SECONDS.
        Outputs: returns self.vector_store once every batch is uploaded.
        Dependencies: calls backend.batching_utils.split_into_batches(),
            upload_batch_with_retry().
        Utilities: called by run_ingestion().
        """
        batches = split_into_batches(chunks, UPLOAD_BATCH_SIZE)
        total_batches = len(batches)

        for batch_number, batch in enumerate(batches, start=1):
            print(f"Uploading batch {batch_number} of {total_batches} ({len(batch)} chunks)...")
            self.upload_batch_with_retry(batch, batch_number, total_batches)

            if batch_number < total_batches:
                print(f"   Micro-batch cooling: Sleeping for {BATCH_PAUSE_SECONDS} seconds...")
                time.sleep(BATCH_PAUSE_SECONDS)

        return self.vector_store


def run_ingestion():
    """
    Description: Main entry point for the production ingestion pipeline -
        scrape, chunk, then upload. Each step is its own small, testable
        function/class above; this function just wires them together
        in order.
    Inputs: none. Reads global config.PINECONE_INDEX_NAME.
    Outputs: no return value.
    Dependencies: calls _scrape_raw_documents(), _has_usable_content(),
        backend.chunking_utils.chunk_documents(), get_embeddings(),
        PineconeBatchUploader.
    Utilities: called by the `if __name__ == "__main__"` block (i.e. this
        is the script's own entry point when run directly).
    """
    raw_docs = _scrape_raw_documents()

    if not _has_usable_content(raw_docs):
        print("ERROR: Web scraper returned empty pages. Ingestion halted.")
        return

    chunks = chunk_documents(raw_docs)
    if not chunks:
        print("ERROR: Generated 0 text chunks. Halting pipeline.")
        return

    print(f"Total chunks generated: {len(chunks)}")
    print("Streaming data up to Pinecone Cloud Indexes in micro-batches...")

    # Uses the SAME shared embeddings provider as vector_store.py and
    # chain_builder.py, instead of creating its own separate instance.
    embeddings = get_embeddings()
    uploader = PineconeBatchUploader(embeddings, config.PINECONE_INDEX_NAME)
    uploader.upload_all_batches(chunks)

    print("Ingestion Complete! Cloud Vector DB is fully up to date without quota errors.")


if __name__ == "__main__":
    run_ingestion()