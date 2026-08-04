import importlib
import os


def test_build_retriever_uses_pinecone_without_local_cache(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "dummy-key")
    monkeypatch.delenv("APP_ENV", raising=False)

    import config

    config = importlib.reload(config)

    import backend.chain_builder as chain_builder

    chain_builder = importlib.reload(chain_builder)

    class DummyStore:
        def as_retriever(self, search_kwargs):
            return {"search_kwargs": search_kwargs, "used": True}

    captured = {}

    def fake_pinecone_vector_store(index_name, embedding):
        captured["index_name"] = index_name
        captured["embedding"] = embedding
        return DummyStore()

    monkeypatch.setattr(chain_builder, "get_embeddings", lambda: object())
    monkeypatch.setattr(
        chain_builder,
        "PineconeVectorStore",
        fake_pinecone_vector_store,
    )

    retriever = chain_builder._build_retriever_from_existing_store()

    assert retriever["used"] is True
    assert captured["index_name"] == config.PINECONE_INDEX_NAME
    assert "embedding" in captured
