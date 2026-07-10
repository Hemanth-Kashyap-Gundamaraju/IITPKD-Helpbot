import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def build_rag_chain(retriever=None):
    """
    Builds the RAG chain. Uses the passed retriever if available, 
    otherwise falls back to connecting directly to the permanent Pinecone index.
    """
    # Fallback to Pinecone if no retriever is passed (e.g., in production/deployment)
    if retriever is None:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        # Retrieve directly from cloud index instantly (Takes 0ms local CPU load)
        vector_db = PineconeVectorStore(index_name="iit-pkd-index", embedding=embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    template = """
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Keep responses clear and formatted appropriately for a mobile chat screen.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )