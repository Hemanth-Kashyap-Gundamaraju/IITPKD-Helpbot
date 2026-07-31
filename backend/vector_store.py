from langchain_pinecone import PineconeVectorStore
import config
from backend.utils.embeddings_utils import get_embeddings
from backend.utils.chunking_utils import chunk_documents


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
    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
    )


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
        print("Error: No scraped documents were returned, so vector ingestion was skipped.")
        return None

    chunks = chunk_documents(documents)
    print(f"Generated {len(chunks)} text chunks. Mapping vectors via Google API...")

    if not chunks:
        print("Error: Generated 0 text chunks. Ingestion halted to prevent database wipe.")
        return None

    embeddings = get_embeddings()
    vector_db = _build_vector_db(chunks, embeddings)

    return vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})