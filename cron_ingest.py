import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from backend import scraper
import config

load_dotenv()

def run_ingestion():
    print("🚀 Starting Production Data Ingestion Pipeline...")
    
    # 1. Scrape only when explicitly updating
    raw_docs = scraper.scrape_target_pages()
    
    # 2. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(raw_docs)
    
    # 3. Stream data up to Pinecone Cloud Indexes
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    print("Uploading elements to Pinecone server...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name="iit-pkd-index"  # Create an index named this in pinecone console
    )
    print("Ingestion Complete! Cloud Vector DB is up to date.")

if __name__ == "__main__":
    run_ingestion()