import time
from pinecone import Pinecone, ServerlessSpec
import config

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# How many times to check if a brand-new Pinecone index has finished
# spinning up, and how long to wait between checks.
INDEX_READY_CHECK_ATTEMPTS = 30
INDEX_READY_CHECK_SLEEP_SECONDS = 2


def _find_matching_index_name(client, embeddings):
    """
    Description: Works out which Pinecone index name matches the current
        embedding model's vector size (a Google embedding and a local
        HuggingFace embedding produce different-sized vectors, so they
        can't share an index). Split out from _get_or_create_index() so
        that function stays short and only handles the "create if
        missing" part.
    Inputs: client (Pinecone client), embeddings (embeddings model).
        Reads global config.PINECONE_INDEX_NAME.
    Outputs: returns (index_name or None, dimension). No globals changed.
    Dependencies: calls embeddings.embed_query().
    Utilities: called by _get_or_create_index().
    """
    sample_vector = embeddings.embed_query("warmup")
    dimension = len(sample_vector)
    target_name = config.PINECONE_INDEX_NAME
    preferred_name = f"{target_name}-{dimension}"

    existing_indexes = {item["name"] for item in client.list_indexes()}

    if preferred_name in existing_indexes:
        index_info = client.describe_index(preferred_name)
        if index_info.get("dimension") == dimension:
            return preferred_name, dimension

    if target_name in existing_indexes:
        index_info = client.describe_index(target_name)
        if index_info.get("dimension") == dimension:
            return target_name, dimension

    return None, dimension


def _wait_until_index_ready(client, index_name):
    """
    Description: Polls Pinecone until a freshly created index reports
        itself as "Ready", so we don't try to write to it too early.
    Inputs: client (Pinecone client), index_name (string). Reads globals
        INDEX_READY_CHECK_ATTEMPTS, INDEX_READY_CHECK_SLEEP_SECONDS.
    Outputs: no return value.
    Dependencies: none.
    Utilities: called by _get_or_create_index().
    """
    for _ in range(INDEX_READY_CHECK_ATTEMPTS):
        try:
            index_info = client.describe_index(index_name)
            if index_info.get("status", {}).get("state") == "Ready":
                return
        except Exception:
            pass
        time.sleep(INDEX_READY_CHECK_SLEEP_SECONDS)


def get_or_create_index(embeddings):
    """
    Description: Finds the Pinecone index that matches the current
        embedding model, or creates one if it doesn't exist yet. This is
        the ONE shared function both the ingestion side (uploading new
        chunks) and the query side (the live server) need - keeping it in
        its own small file means the server never has to import the
        ingestion file (vector_store.py) just to get this one function.
    Inputs: embeddings (the embeddings model in use). Reads global
        config.PINECONE_API_KEY.
    Outputs: returns the index name (string). May create a new index in
        Pinecone as a side effect.
    Dependencies: calls _find_matching_index_name(),
        _wait_until_index_ready(); uses pinecone.Pinecone,
        pinecone.ServerlessSpec.
    Utilities: called by backend.vector_store._build_vector_db() and
        backend.chain_builder._build_retriever_from_existing_store().
    """
    client = Pinecone(api_key=config.PINECONE_API_KEY)
    matching_name, dimension = _find_matching_index_name(client, embeddings)

    if matching_name:
        return matching_name

    preferred_name = f"{config.PINECONE_INDEX_NAME}-{dimension}"
    client.create_index(
        name=preferred_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    _wait_until_index_ready(client, preferred_name)
    return preferred_name