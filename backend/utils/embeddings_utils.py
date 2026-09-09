from langchain_google_genai import GoogleGenerativeAIEmbeddings
import config


class EmbeddingsProvider:
    """
    Description: Builds and caches the Google Gemini embeddings model.
        This is the ONLY place in the whole project that should create an
        embeddings object - everything else goes through get_embeddings()
        below. Wrapped in a class (instead of a bare module-level cache
        variable) so a test can create a fresh EmbeddingsProvider() and
        swap in fake settings without needing to reset any global state.
    Inputs (constructor): none.
    Utilities: used by get_embeddings().
    """

    def __init__(self):
        self._cached_embeddings = None

    def _create_google_embeddings(self):
        """
        Description: Builds one fresh Google Gemini embeddings object,
            using the API key and model name from config. Split out into
            its own small helper so get() only has to worry about caching
            - not about how the embeddings object itself gets built.
        Inputs: none. Reads global config.GOOGLE_API_KEY,
            config.EMBEDDING_MODEL_NAME.
        Outputs: returns a GoogleGenerativeAIEmbeddings object. No globals
            changed.
        Dependencies: uses
            langchain_google_genai.GoogleGenerativeAIEmbeddings.
        Utilities: called by get().
        """
        print(f"Using GOOGLE embeddings ({config.EMBEDDING_MODEL_NAME}).")
        return GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY,
        )

    def get(self):
        """
        Description: Returns the embeddings model, creating it on the
            first call and reusing it after that.
        Inputs: none. Reads self._cached_embeddings.
        Outputs: returns an embeddings object. Sets
            self._cached_embeddings (on the first call).
        Dependencies: calls _create_google_embeddings().
        Utilities: called by get_embeddings().
        """
        if self._cached_embeddings is not None:
            return self._cached_embeddings

        self._cached_embeddings = self._create_google_embeddings()
        return self._cached_embeddings


# One shared instance for the whole app, so every caller reuses the same
# cached embeddings model instead of each creating its own.
_provider = EmbeddingsProvider()


def get_embeddings():
    """
    Description: Returns the shared embeddings model. Thin function kept
        here so every other module can keep doing
        `from backend.utils.embeddings_utils import get_embeddings`
        without needing to know about the EmbeddingsProvider class
        underneath.
    Inputs: none. Reads global _provider.
    Outputs: returns an embeddings object. No globals changed (mutation
        happens inside _provider itself).
    Dependencies: calls EmbeddingsProvider.get().
    Utilities: called by backend.vector_store, backend.chain_builder,
        update_database.py.
    """
    return _provider.get()