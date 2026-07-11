import os
import shutil
import random
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import config


def initialize_vector_db(documents):
    """Chunks scraped text documents and builds the vector database."""
    if not documents:
        print(
            "❌ Error: No scraped documents were returned, so vector ingestion was skipped."
        )
        return None

    # Clear old cache directory to prevent database dimensions collisions
    if (
        config.APP_ENV == "local"
        and config.CACHE_DIR
        and os.path.exists(config.CACHE_DIR)
    ):
        print("Clearing old vector cache...")
        shutil.rmtree(config.CACHE_DIR)

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

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    if config.APP_ENV == "production":
        vector_db = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=config.PINECONE_INDEX_NAME,
        )
        return vector_db.as_retriever(search_kwargs={"k": 3})

    # Resilient DB assignment tracking to overcome heavy server traffic
    vector_db = None
    for attempt in range(4):
        try:
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=config.CACHE_DIR,
            )
            break
        except Exception as e:
            if "503" in str(e) and attempt < 3:
                sleep_time = 4 + random.uniform(1, 3)
                print(
                    f"   [Server Busy] Embedding engine congested. Retrying database upload in {sleep_time:.1f}s..."
                )
                time.sleep(sleep_time)
            else:
                print(f"\n[Error during vectorization]: {e}")
                exit()

    return vector_db.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    """Formats context documents into an ordered block string format."""
    return "\n\n".join(doc.page_content for doc in docs)
