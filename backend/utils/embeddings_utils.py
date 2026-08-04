from langchain_huggingface import HuggingFaceEmbeddings
import config


class EmbeddingsProvider:
    """
    Description: Builds and caches the embeddings model to use, based on
        config.EMBEDDING_PROVIDER. This is the ONLY place in the whole
        project that should create an embeddings object - everything
        else goes through get_embeddings() below. Wrapped in a class
        (instead of a bare module-level cache variable) so a test can
        create a fresh EmbeddingsProvider() and swap in fake settings
        without needing to reset any global state.
    Inputs (constructor): none.
    Utilities: used by get_embeddings().
    """

    def __init__(self):
        self._cached_embeddings = None

    def get(self):
        """
        Description: Returns the embeddings model, creating it on the
            first call and reusing it after that.
        Inputs: none. Reads self._cached_embeddings, global
            config.EMBEDDING_PROVIDER, config.LOCAL_EMBEDDING_MODEL_NAME,
            config.EMBEDDING_MODEL_NAME.
        Outputs: returns an embeddings object. Sets self._cached_embeddings
            (on the first call).
        Dependencies: uses langchain_huggingface.HuggingFaceEmbeddings,
            langchain_google_genai.GoogleGenerativeAIEmbeddings.
        Utilities: called by get_embeddings().
        """
        if self._cached_embeddings is not None:
            return self._cached_embeddings


        print(f"Using LOCAL embeddings ({config.LOCAL_EMBEDDING_MODEL_NAME}) — no quota limits.")
        self._cached_embeddings = HuggingFaceEmbeddings(
                model_name=config.LOCAL_EMBEDDING_MODEL_NAME
            )


        return self._cached_embeddings


# One shared instance for the whole app, so every caller reuses the same
# cached embeddings model instead of each creating its own.
_provider = EmbeddingsProvider()


def get_embeddings():
    """
    Description: Returns the shared embeddings model. Thin function kept
        here so every other module can keep doing
        `from backend.embeddings_utils import get_embeddings` without
        needing to know about the EmbeddingsProvider class underneath.
    Inputs: none. Reads global _provider.
    Outputs: returns an embeddings object. No globals changed (mutation
        happens inside _provider itself).
    Dependencies: calls EmbeddingsProvider.get().
    Utilities: called by backend.vector_store, backend.chain_builder,
        cron_ingest.
    """
    return _provider.get()