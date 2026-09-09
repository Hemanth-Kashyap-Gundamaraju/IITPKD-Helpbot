from langchain_pinecone import PineconeVectorStore
import config
import time
from backend.utils.embeddings_utils import get_embeddings
from backend.utils.chunking_utils import chunk_documents
from backend.utils.pinecone_index_utils import get_or_create_index


def _build_vector_db(chunks, embeddings):
    """
    Description: Builds (or connects to) the Pinecone index and uploads
        every chunk in one call - Pinecone's own client handles batching
        server-side, so no manual batch loop is needed here.
    Inputs: chunks (full list of chunked Documents), embeddings (the
        embeddings model to use). Reads global config.PINECONE_INDEX_NAME.
    Outputs: returns the PineconeVectorStore object. No globals changed.
    Dependencies: uses langchain_pinecone.PineconeVectorStore.
    Utilities: called by initialize_vector_db().
    """
    index_name = get_or_create_index(embeddings)
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Uploading batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}...")
        vector_store.add_documents(batch)
        if i + batch_size < len(chunks):
            print("Pausing for 65s to respect Google GenAI 100-req/min free tier rate limits...")
            time.sleep(65)
            
    return vector_store


def initialize_vector_db(documents):
    """
    Description: Chunks scraped text documents and uploads them to the
        Pinecone vector database.
    Inputs: documents (list of raw Document objects). Reads global
        config.RETRIEVER_TOP_K.
    Outputs: returns a retriever object, or None if ingestion was skipped.
    Dependencies: calls backend.utils.chunking_utils.chunk_documents(),
        get_embeddings(), _build_vector_db().
    Utilities: called by main.py (via backend.vector_store.initialize_vector_db).
    """
    if not documents:
        print(
            "Error: No scraped documents were returned, so vector ingestion was skipped."
        )
        return None

    chunks = chunk_documents(documents)
    print(f"Generated {len(chunks)} text chunks. Mapping vectors via Google API...")

    if not chunks:
        print(
            "Error: Generated 0 text chunks. Ingestion halted to prevent database wipe."
        )
        return None

    embeddings = get_embeddings()
    vector_db = _build_vector_db(chunks, embeddings)

    return vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})
