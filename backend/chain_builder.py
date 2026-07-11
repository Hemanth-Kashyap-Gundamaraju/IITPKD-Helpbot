from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import config


def build_rag_chain(retriever=None):
    """
    Builds the RAG chain. Uses the passed retriever if available,
    otherwise selects the vector store based on APP_ENV.
    """
    if retriever is None:
        embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
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
        retriever = vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})

    template = """
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Keep responses clear and formatted appropriately for a mobile chat screen.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    llm = ChatGroq(model=config.LLM_MODEL_NAME, temperature=config.LLM_TEMPERATURE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
