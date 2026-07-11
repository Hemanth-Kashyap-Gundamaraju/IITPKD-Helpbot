# --- MUST PLACE THIS AT THE ABSOLUTE TOP OF THE FILE ---
import os

os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# ------------------------------------------------------

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from backend import scraper
import config

load_dotenv()


def run_ingestion():
    print(f"🚀 Starting {config.APP_ENV.title()} Data Ingestion Pipeline...")

    raw_docs = scraper.scrape_target_pages()

    # CRITICAL SERVER FALLBACK: Check if pages returned completely empty text
    # to avoid empty list crashes during Pinecone upserting
    if not raw_docs or all(len(doc.page_content.strip()) == 0 for doc in raw_docs):
        print(
            "❌ Error: Web crawler was blocked or returned empty content pages from this server IP context."
        )
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(raw_docs)

    if not chunks:
        print(
            "❌ Error: Generated 0 text chunks. Ingestion halted to prevent database wipe."
        )
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    if config.APP_ENV == "production":
        print(f"Uploading {len(chunks)} elements to Pinecone server...")
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=config.PINECONE_INDEX_NAME,
        )
        print("Ingestion Complete! Cloud Vector DB is up to date.")
        return

    print(
        f"Uploading {len(chunks)} elements to local Chroma cache at {config.CACHE_DIR}..."
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CACHE_DIR,
    )
    print("Ingestion Complete! Local vector DB is up to date.")


if __name__ == "__main__":
    run_ingestion()
