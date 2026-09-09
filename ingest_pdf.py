import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load env variables so API keys are present
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

import config
from backend.utils.embeddings_utils import get_embeddings
from backend.utils.pinecone_index_utils import get_or_create_index
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader

def ingest_pdf(pdf_path_or_url):
    print(f"Loading PDF from: {pdf_path_or_url}...")
    try:
        loader = PyPDFLoader(pdf_path_or_url)
        documents = loader.load()
    except Exception as e:
        print(f"Failed to load PDF: {e}")
        sys.exit(1)

    print(f"Loaded {len(documents)} pages from PDF.")

    # We do not chunk the PDF aggressively because tables might break
    # We will upload each page as a single chunk, which works well for schedules
    print("Connecting to Pinecone database...")
    embeddings = get_embeddings()
    index_name = get_or_create_index(embeddings)
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    print("Uploading PDF pages to Pinecone...")
    vector_store.add_documents(documents)
    print("PDF successfully ingested! The bot now knows the bus timings.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdf.py <URL_or_path_to_pdf>")
        sys.exit(1)
        
    pdf_target = sys.argv[1]
    ingest_pdf(pdf_target)
