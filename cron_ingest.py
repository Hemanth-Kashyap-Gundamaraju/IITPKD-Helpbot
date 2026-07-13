# =====================================================================
# CRITICAL: This environment patch MUST run before any other imports!
# =====================================================================
import os
from dotenv import load_dotenv

load_dotenv()

import config

os.environ["USER_AGENT"] = config.SCRAPER_HEADERS["User-Agent"]
# =====================================================================

import time  # <--- Added for rate-limit throttling
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from backend import scraper



def run_ingestion():
    print("Starting Production Data Ingestion Pipeline...")

    # 1. Scrape target pages dynamically
    raw_docs = scraper.scrape_target_pages()

    # SERVER SAFEGUARD: Check if the university server blocked us or returned blank text
    if not raw_docs or all(len(doc.page_content.strip()) == 0 for doc in raw_docs):
        print("❌ ERROR: Web scraper returned empty pages. Ingestion halted.")
        return

    # 2. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(raw_docs)

    if not chunks:
        print(" ERROR: Generated 0 text chunks. Halting pipeline.")
        return

    # 3. Initialize Embeddings Engine
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
    index_name = config.PINECONE_INDEX_NAME

    print(f"Total chunks generated: {len(chunks)}")
    print(" Streaming data up to Pinecone Cloud Indexes in micro-batches...")

    # 4. Ultra-Safe Micro-Batching Loop
    BATCH_SIZE = config.INGESTION_BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(
            f"Uploading batch {(i // BATCH_SIZE) + 1} of {(len(chunks) // BATCH_SIZE) + 1} ({len(batch)} chunks)..."
        )

        try:
            if i == 0:
                vector_store = PineconeVectorStore.from_documents(
                    documents=batch, embedding=embeddings, index_name=index_name
                )
            else:
                vector_store.add_documents(documents=batch)

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(" Hit Gemini Rate Limit! Enforcing emergency cooldown...")
                time.sleep(config.INGESTION_RATE_LIMIT_COOLDOWN)
                # Retry the exact same batch
                if i == 0:
                    vector_store = PineconeVectorStore.from_documents(
                        documents=batch, embedding=embeddings, index_name=index_name
                    )
                else:
                    vector_store.add_documents(documents=batch)
            else:
                raise e

        # Standard cool-down delay between every micro-batch
        if i + BATCH_SIZE < len(chunks):
            print(
                f" Micro-batch cooling: Sleeping for {config.INGESTION_BATCH_PAUSE} seconds..."
            )
            time.sleep(config.INGESTION_BATCH_PAUSE)

    print(
        "Ingestion Complete! Cloud Vector DB is fully up to date without quota errors."
    )
