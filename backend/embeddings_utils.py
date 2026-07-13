# backend/embeddings_utils.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import config

# Keep one instance so we don't reload the local model on every call
_cached_embeddings = None


def get_embeddings():
    """
    Returns the embedding model to use, based on config.EMBEDDING_PROVIDER.
    This is the ONLY place in the whole project that should create an
    embeddings object. Everything else calls this function.
    """
    global _cached_embeddings

    if _cached_embeddings is not None:
        return _cached_embeddings

    if config.EMBEDDING_PROVIDER == "local":
        print(f"Using LOCAL embeddings ({config.LOCAL_EMBEDDING_MODEL_NAME}) — no quota limits.")
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL_NAME
        )
    else:
        print(f"Using GOOGLE embeddings ({config.EMBEDDING_MODEL_NAME}) — quota applies.")
        _cached_embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME
        )

    return _cached_embeddings