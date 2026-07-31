from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def chunk_documents(raw_docs):
    """
    Description: Splits scraped documents into smaller text chunks ready
        for embedding. Pure transformation - same input always gives the
        same output. Shared by every ingestion path (local Chroma and
        production Pinecone) so the chunk size/overlap settings only
        need to be tuned in one place.
    Inputs: raw_docs (list of Document objects). Reads globals
        config.CHUNK_SIZE, config.CHUNK_OVERLAP.
    Outputs: returns a list of chunked Document objects. No globals changed.
    Dependencies: uses langchain_text_splitters.RecursiveCharacterTextSplitter.
    Utilities: called by backend.vector_store.initialize_vector_db() and
        cron_ingest.run_ingestion().
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return text_splitter.split_documents(raw_docs)
