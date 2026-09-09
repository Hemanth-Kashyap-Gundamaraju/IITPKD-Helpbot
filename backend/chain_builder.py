from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from backend.utils.embeddings_utils import get_embeddings
import config
from backend.utils.pinecone_index_utils import get_or_create_index
from datetime import datetime

def get_current_time():
    return datetime.now().strftime("%A, %Y-%m-%d %I:%M %p")

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# The exact sentence the LLM must say when the context doesn't answer the
# question. Kept as a global constant (not hardcoded inline in the prompt)
# so if we ever want to change this wording, there's only ONE place to edit.
NO_ANSWER_FALLBACK_MESSAGE = "I don't have verified information about that from the IIT Palakkad website."

RAG_PROMPT_TEMPLATE = f"""
You are the official AI Helpbot for IIT Palakkad (Indian Institute of Technology Palakkad).
Use the following pieces of retrieved context to answer the user's question. 
Follow these rules strictly:
1. For factual queries (e.g. bus timings, faculty names, fees, deadlines), NEVER use outside information. If the answer is not in the context, explicitly say: "{NO_ANSWER_FALLBACK_MESSAGE}"
2. For subjective queries or general student advice (e.g. comparing CS vs DS, career guidance, study tips), you MAY use your general knowledge to provide a helpful, conversational answer.
3. Keep responses clear, concise, and formatted appropriately for a mobile WhatsApp screen.
4. If you used the retrieved context to answer a factual question, ALWAYS cite the source URLs at the end in a 'Sources:' section. If you gave general advice or didn't know the answer, DO NOT include a 'Sources:' section.
5. If the user asks a question that is clearly unrelated to academics, campus life, or engineering (e.g. "how do I cook pasta"), refuse to answer and state that you are an IIT Palakkad Helpbot.

The current date and time is: {{current_time}}. You can use this to answer time-relative questions like "next bus" based on the provided schedules.

Recent Chat History:
{{chat_history}}

Context:
{{context}}

Question: {{question}}

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
    docs_text = []
    clean_count = 0
    for doc in docs:
        content = doc.page_content
        source = doc.metadata.get("source", "Unknown Source")
        
        # Hard-filter the recurring junk navigation chunks (skip filter for PDFs)
        if not source.lower().endswith(".pdf"):
            if content.count('\n') > 15 or "Englishहिन्दी" in content:
                continue
        docs_text.append(f"Content:\n{content}\nSource: {source}")
        
        # Only take the top 7 clean chunks
        clean_count += 1
        if clean_count >= 7:
            break
            
    return "\n\n---\n\n".join(docs_text)


def _build_retriever_from_existing_store():
    """
    Description: Connects to the configured Pinecone index and wraps it as a
        retriever for the RAG chain, without falling back to a local cache or
        creating a new index on startup.
    Inputs: none. Reads global config.RETRIEVER_TOP_K.
    Outputs: returns a retriever object. No globals changed.
    Dependencies: calls get_embeddings(), get_or_create_index(); uses
        langchain_pinecone.PineconeVectorStore.
    Utilities: called by build_rag_chain().
    """
    embeddings = get_embeddings()
    index_name = get_or_create_index(embeddings)
    vector_db = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )
    return vector_db.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 250}
    )

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
    Utilities: called by main.py and asgi.py.
    """
    if retriever is None:
        retriever = _build_retriever_from_existing_store()

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question", "chat_history"],
        partial_variables={"current_time": get_current_time}
    )
    llm = ChatGroq(model=config.LLM_MODEL_NAME, temperature=config.LLM_TEMPERATURE, max_tokens=4000)

    return (
        {
            "context": itemgetter("question") | retriever | format_docs, 
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
    )
