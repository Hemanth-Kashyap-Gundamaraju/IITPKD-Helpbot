from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
import backend.vector_store as vector_store

def build_rag_chain(retriever):
    """Configures the template prompt, links the LLM, and returns the compiled RAG chain."""
    template = """
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    # Initialize Free Groq Engine
    llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0)

    # Return the assembled declarative LangChain Expression Language (LCEL) chain
    return (
        {"context": retriever | vector_store.format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )