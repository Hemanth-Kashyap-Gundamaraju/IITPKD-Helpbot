from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import config
from backend.utils.embeddings_utils import get_embeddings
from backend.utils.chunking_utils import chunk_documents
from backend.utils.pinecone_index_utils import get_or_create_index

def _get_or_create_index(embeddings):
    """Create or select a Pinecone index compatible with the active embedding model."""
    client = Pinecone(api_key=config.PINECONE_API_KEY)
    sample_vector = embeddings.embed_query("warmup")
    dimension = len(sample_vector)
    target_name = config.PINECONE_INDEX_NAME
    dimension_suffix = f"-{dimension}"
    preferred_name = f"{target_name}{dimension_suffix}"

    existing_indexes = {item["name"] for item in client.list_indexes()}

    if preferred_name in existing_indexes:
        index_info = client.describe_index(preferred_name)
        if index_info.get("dimension") == dimension:
            return preferred_name

    if target_name in existing_indexes:
        index_info = client.describe_index(target_name)
        if index_info.get("dimension") == dimension:
            return target_name

    if preferred_name in existing_indexes:
        return preferred_name

    client.create_index(
        name=preferred_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    import time

    for _ in range(30):
        try:
            index_info = client.describe_index(preferred_name)
            if index_info.get("status", {}).get("state") == "Ready":
                break
        except Exception:
            pass
        time.sleep(2)

    return preferred_name


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
    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
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
