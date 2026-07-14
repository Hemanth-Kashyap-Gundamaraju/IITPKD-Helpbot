import os
import shutil
import random
import time
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
import config
from backend.embeddings_utils import get_embeddings
from backend.chunking_utils import chunk_documents
from backend.batching_utils import split_into_batches

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# How many chunks we send to Google in one embedding request when building
# the LOCAL Chroma database. Sending too many at once is what triggers the
# 429 "quota exceeded" error.
LOCAL_EMBED_BATCH_SIZE = config.INGESTION_BATCH_SIZE

# How long we wait between two normal, successful batches (to avoid
# tripping the per-minute limit in the first place).
LOCAL_EMBED_BATCH_PAUSE = config.INGESTION_BATCH_PAUSE

# How long we wait after Google tells us "you're over quota, slow down".
LOCAL_EMBED_RATE_LIMIT_COOLDOWN = config.INGESTION_RATE_LIMIT_COOLDOWN


def _is_quota_error(error):
    """
    Description: Checks if an error is a '429 RESOURCE_EXHAUSTED' quota error.
    Inputs: error (the exception object). No globals read.
    Outputs: returns True/False. No globals changed.
    Dependencies: none.
    Utilities: called by LocalChromaIngestor.add_batch_with_retries().
    """
    error_text = str(error)
    return "429" in error_text or "RESOURCE_EXHAUSTED" in error_text


def _is_server_busy_error(error):
    """
    Description: Checks if an error is a '503' error, meaning Google's
        servers are temporarily overloaded (different problem than
        running out of quota).
    Inputs: error (the exception object). No globals read.
    Outputs: returns True/False. No globals changed.
    Dependencies: none.
    Utilities: called by LocalChromaIngestor.add_batch_with_retries().
    """
    return "503" in str(error)


def _get_backoff_sleep_seconds():
    """
    Description: Calculates how long to wait before retrying after a 503
        (server busy) error, with a bit of random jitter so retries from
        multiple batches don't all land at the same instant.
    Inputs: none. Reads globals config.VECTOR_DB_RETRY_SLEEP_BASE,
        config.VECTOR_DB_RETRY_SLEEP_JITTER_MIN, config.VECTOR_DB_RETRY_SLEEP_JITTER_MAX.
    Outputs: returns a float (seconds to sleep). No globals changed.
    Dependencies: none.
    Utilities: called by LocalChromaIngestor.add_batch_with_retries().
    """
    return config.VECTOR_DB_RETRY_SLEEP_BASE + random.uniform(
        config.VECTOR_DB_RETRY_SLEEP_JITTER_MIN,
        config.VECTOR_DB_RETRY_SLEEP_JITTER_MAX,
    )


def _clear_old_local_cache():
    """
    Description: Deletes the old local Chroma folder so we don't mix old
        and new data when re-ingesting from scratch.
    Inputs: none. Reads globals config.APP_ENV, config.CACHE_DIR.
    Outputs: no return value. Deletes the folder at config.CACHE_DIR on disk
        (not a Python global, but a real side effect worth flagging).
    Dependencies: none.
    Utilities: called by initialize_vector_db().
    """
    if (
        config.APP_ENV == "local"
        and config.CACHE_DIR
        and os.path.exists(config.CACHE_DIR)
    ):
        print("Clearing old vector cache...")
        shutil.rmtree(config.CACHE_DIR)


class LocalChromaIngestor:
    """
    Description: Handles embedding and writing chunk batches into the local
        Chroma database, with retries on rate-limit/server-busy errors.
        Groups together the embeddings model and the growing Chroma
        connection - the two things every step needs - as attributes on
        one object instead of passing them separately into every function.
    Inputs (constructor): embeddings (the embeddings model to use).
    Utilities: used by initialize_vector_db().
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_db = None  # set once the first batch creates the local DB

    def add_batch(self, batch):
        """
        Description: Adds one batch of chunks to the local Chroma database.
            Creates the database on the first call, adds to it on every
            call after that. This is the only method that actually writes
            to disk/calls the embeddings API, so it's the only piece that
            needs a real (or mocked) connection to test.
        Inputs: batch (list of Document chunks). Reads self.embeddings,
            self.vector_db. Reads global config.CACHE_DIR.
        Outputs: no return value. Sets self.vector_db (on the first call).
        Dependencies: uses langchain_community.vectorstores.Chroma.
        Utilities: called by add_batch_with_retries().
        """
        if self.vector_db is None:
            self.vector_db = Chroma.from_documents(
                documents=batch,
                embedding=self.embeddings,
                persist_directory=config.CACHE_DIR,
            )
        else:
            self.vector_db.add_documents(batch)

    def add_batch_with_retries(self, batch, batch_number, total_batches):
        """
        Description: Tries to embed and store one batch of chunks, retrying
            automatically on a 429 (quota) or 503 (server busy) error.
            Gives up and exits the whole program only after too many
            failed attempts.
        Inputs: batch, batch_number, total_batches. Reads global
            config.VECTOR_DB_RETRY_ATTEMPTS, LOCAL_EMBED_RATE_LIMIT_COOLDOWN.
        Outputs: no return value (self.vector_db is updated via add_batch).
        Dependencies: calls add_batch(), _is_quota_error(),
            _is_server_busy_error(), _get_backoff_sleep_seconds().
        Utilities: called by ingest_all_chunks().
        """
        for attempt in range(config.VECTOR_DB_RETRY_ATTEMPTS):
            try:
                self.add_batch(batch)
                return
            except Exception as e:
                is_last_attempt = attempt == config.VECTOR_DB_RETRY_ATTEMPTS - 1

                if _is_quota_error(e) and not is_last_attempt:
                    print(
                        f"   [Rate Limit] Batch {batch_number}/{total_batches} hit the free-tier "
                        f"quota. Cooling down for {LOCAL_EMBED_RATE_LIMIT_COOLDOWN}s..."
                    )
                    time.sleep(LOCAL_EMBED_RATE_LIMIT_COOLDOWN)

                elif _is_server_busy_error(e) and not is_last_attempt:
                    sleep_time = _get_backoff_sleep_seconds()
                    print(
                        f"   [Server Busy] Batch {batch_number}/{total_batches} congested. "
                        f"Retrying in {sleep_time:.1f}s..."
                    )
                    time.sleep(sleep_time)

                else:
                    print(f"\n[Error during vectorization]: {e}")
                    exit()

    def ingest_all_chunks(self, chunks):
        """
        Description: Embeds and stores every chunk into the local Chroma
            database, batch by batch, pausing between batches to stay
            under the rate limit. Orchestrates the other methods on this
            class rather than doing any single thing itself.
        Inputs: chunks (full list of chunked Documents). Reads global
            LOCAL_EMBED_BATCH_SIZE, LOCAL_EMBED_BATCH_PAUSE.
        Outputs: returns self.vector_db once every batch is stored.
        Dependencies: calls backend.batching_utils.split_into_batches(),
            add_batch_with_retries().
        Utilities: called by initialize_vector_db().
        """
        batches = split_into_batches(chunks, LOCAL_EMBED_BATCH_SIZE)
        total_batches = len(batches)

        for batch_number, batch in enumerate(batches, start=1):
            print(f"Embedding batch {batch_number}/{total_batches} ({len(batch)} chunks)...")
            self.add_batch_with_retries(batch, batch_number, total_batches)

            if batch_number < total_batches:
                time.sleep(LOCAL_EMBED_BATCH_PAUSE)

        return self.vector_db


def _build_production_vector_db(chunks, embeddings):
    """
    Description: Builds (or connects to) the production Pinecone index and
        uploads every chunk in one call - Pinecone's own client handles
        batching server-side, so no manual batch loop is needed here.
    Inputs: chunks (full list of chunked Documents), embeddings (the
        embeddings model to use). Reads global config.PINECONE_INDEX_NAME.
    Outputs: returns the PineconeVectorStore object. No globals changed.
    Dependencies: uses langchain_pinecone.PineconeVectorStore.
    Utilities: called by initialize_vector_db().
    """
    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
    )


def initialize_vector_db(documents):
    """
    Description: Chunks scraped text documents and builds the vector
        database - production Pinecone or local Chroma, depending on
        config.APP_ENV.
    Inputs: documents (list of raw Document objects). Reads global
        config.APP_ENV, config.RETRIEVER_TOP_K.
    Outputs: returns a retriever object, or None if ingestion was skipped.
    Dependencies: calls _clear_old_local_cache(),
        backend.chunking_utils.chunk_documents(), get_embeddings(),
        _build_production_vector_db(), LocalChromaIngestor.
    Utilities: called by main.py (via backend.vector_store.initialize_vector_db).
    """
    if not documents:
        print("Error: No scraped documents were returned, so vector ingestion was skipped.")
        return None

    _clear_old_local_cache()

    chunks = chunk_documents(documents)
    print(f"Generated {len(chunks)} text chunks. Mapping vectors via Google API...")

    if not chunks:
        print("Error: Generated 0 text chunks. Ingestion halted to prevent database wipe.")
        return None

    embeddings = get_embeddings()

    if config.APP_ENV == "production":
        vector_db = _build_production_vector_db(chunks, embeddings)
    else:
        ingestor = LocalChromaIngestor(embeddings)
        vector_db = ingestor.ingest_all_chunks(chunks)

    return vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})