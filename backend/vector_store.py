import os
import shutil
import random
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import config

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS (pulled from config so they're easy to tune in one place)
# ---------------------------------------------------------------------------
# How many chunks we send to Google in one embedding request.
# Sending too many at once is what triggers the 429 "quota exceeded" error.
LOCAL_EMBED_BATCH_SIZE = config.INGESTION_BATCH_SIZE

# How long we wait between two normal, successful batches (to avoid
# tripping the per-minute limit in the first place).
LOCAL_EMBED_BATCH_PAUSE = config.INGESTION_BATCH_PAUSE

# How long we wait after Google tells us "you're over quota, slow down".
LOCAL_EMBED_RATE_LIMIT_COOLDOWN = config.INGESTION_RATE_LIMIT_COOLDOWN


def _is_quota_error(error):
    """
    Checks if an error is a '429 RESOURCE_EXHAUSTED' quota error.
    This happens when we send too many embedding requests too fast.
    """
    error_text = str(error)
    return "429" in error_text or "RESOURCE_EXHAUSTED" in error_text


def _is_server_busy_error(error):
    """
    Checks if an error is a '503' error, meaning Google's servers are
    temporarily overloaded (different problem than running out of quota).
    """
    return "503" in str(error)


def _get_backoff_sleep_seconds():
    """Calculates how long to wait before retrying after a 503 (server busy) error."""
    return config.VECTOR_DB_RETRY_SLEEP_BASE + random.uniform(
        config.VECTOR_DB_RETRY_SLEEP_JITTER_MIN,
        config.VECTOR_DB_RETRY_SLEEP_JITTER_MAX,
    )


def _clear_old_local_cache():
    """Deletes the old local Chroma folder so we don't mix old and new data."""
    if (
        config.APP_ENV == "local"
        and config.CACHE_DIR
        and os.path.exists(config.CACHE_DIR)
    ):
        print("Clearing old vector cache...")
        shutil.rmtree(config.CACHE_DIR)


def _split_into_batches(items, batch_size):
    """Splits a list into smaller lists (batches) of the given size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _add_batch_to_local_db(vector_db, batch, embeddings):
    """
    Adds one batch of chunks to the local Chroma database.
    If the database doesn't exist yet, this creates it.
    If it already exists, this just adds more documents to it.
    """
    if vector_db is None:
        return Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory=config.CACHE_DIR,
        )
    vector_db.add_documents(batch)
    return vector_db


def _embed_batch_with_retries(vector_db, batch, embeddings, batch_number, total_batches):
    """
    Tries to embed and store one batch of chunks.
    Retries automatically if we hit a 429 (quota) or 503 (server busy) error.
    Gives up and exits only after too many failed attempts.
    """
    for attempt in range(config.VECTOR_DB_RETRY_ATTEMPTS):
        try:
            return _add_batch_to_local_db(vector_db, batch, embeddings)
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


def initialize_vector_db(documents):
    """Chunks scraped text documents and builds the vector database."""
    if not documents:
        print(
            "❌ Error: No scraped documents were returned, so vector ingestion was skipped."
        )
        return None

    _clear_old_local_cache()

    # Split structural elements into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} text chunks. Mapping vectors via Google API...")

    if not chunks:
        print(
            "❌ Error: Generated 0 text chunks. Ingestion halted to prevent database wipe."
        )
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)

    if config.APP_ENV == "production":
        vector_db = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=config.PINECONE_INDEX_NAME,
        )
        return vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})

    # --- Local Chroma path: send chunks in small batches instead of all at once ---
    # This is the actual fix: sending everything in one shot is what triggers
    # the 429 "quota exceeded" error you saw.
    batches = _split_into_batches(chunks, LOCAL_EMBED_BATCH_SIZE)
    total_batches = len(batches)
    vector_db = None

    for batch_number, batch in enumerate(batches, start=1):
        print(f"Embedding batch {batch_number}/{total_batches} ({len(batch)} chunks)...")
        vector_db = _embed_batch_with_retries(
            vector_db, batch, embeddings, batch_number, total_batches
        )

        # Small pause between batches so we don't bump into the per-minute
        # limit again on the very next batch.
        if batch_number < total_batches:
            time.sleep(LOCAL_EMBED_BATCH_PAUSE)

    return vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})


def format_docs(docs):
    """Formats context documents into an ordered block string format."""
    return "\n\n".join(doc.page_content for doc in docs)