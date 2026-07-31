from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from backend.utils.embeddings_utils import get_embeddings

import config

RAG_PROMPT_TEMPLATE = """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep responses clear and formatted appropriately for a mobile chat screen.

Context: {context}

Question: {question}

Answer:
"""


def format_docs(docs):
    """
    Description: Formats retrieved context documents into one text block,
        separated by blank lines, ready to drop into the prompt template.
        Pure transformation - promoted to module level (it used to be
        duplicated as a private nested function inside both this file and
        backend/vector_store.py) so there is only one copy in the project.
    Inputs: docs (list of Document objects). No globals read.
    Outputs: returns a single string. No globals changed.
    Dependencies: none.
    Utilities: called by build_rag_chain().
    """
    return "\n\n".join(doc.page_content for doc in docs)


def _build_retriever_from_existing_store():
    """
    Description: Connects to whichever vector store is configured
        (production Pinecone or local Chroma) and wraps it as a retriever,
        for the case where main.py didn't already build one.
    Inputs: none. Reads global config.APP_ENV, config.PINECONE_INDEX_NAME,
        config.CACHE_DIR, config.RETRIEVER_TOP_K.
    Outputs: returns a retriever object. No globals changed.
    Dependencies: calls get_embeddings(); uses
        langchain_pinecone.PineconeVectorStore, langchain_community.vectorstores.Chroma.
    Utilities: called by build_rag_chain().
    """
    embeddings = get_embeddings()
    if config.APP_ENV == "production":
        vector_db = PineconeVectorStore(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
        )
    else:
        vector_db = Chroma(
            persist_directory=config.CACHE_DIR,
            embedding_function=embeddings,
        )
    return vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})


def build_rag_chain(retriever=None):
    """
    Description: Builds the RAG chain that takes a user question, retrieves
        matching context, and asks the Groq LLM to answer using only that
        context.
    Inputs: retriever (optional; if None, one is built from the configured
        vector store). Reads global config.LLM_MODEL_NAME, config.LLM_TEMPERATURE.
    Outputs: returns a LangChain runnable chain. No globals changed.
    Dependencies: calls _build_retriever_from_existing_store() (only if no
        retriever was passed in), format_docs(); uses langchain_groq.ChatGroq.
    Utilities: called by main.py and wsgi.py.
    """
    if retriever is None:
        retriever = _build_retriever_from_existing_store()

    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    llm = ChatGroq(model=config.LLM_MODEL_NAME, temperature=config.LLM_TEMPERATURE)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )