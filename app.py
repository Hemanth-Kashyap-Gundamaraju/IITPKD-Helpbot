import streamlit as st
import time
from pathlib import Path
from dotenv import load_dotenv

# Load env variables so API keys are present
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from backend.chain_builder import build_rag_chain
from backend.utils.response_utils import clean_llm_response
from backend.utils.cache_utils import get_response_cache
from backend.utils.analytics import get_query_logger
from backend.utils.escalation import get_escalation_handler

# Page setup
st.set_page_config(page_title="IITPKD Helpbot", page_icon="🎓", layout="centered")
st.title("IIT Palakkad Helper Bot 🎓")
st.markdown("Ask me anything about IIT Palakkad! I am powered by RAG and Pinecone vectors.")

# Initialize RAG chain only once and cache it in Streamlit
@st.cache_resource
def get_rag_chain():
    return build_rag_chain(retriever=None)

try:
    rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"Failed to connect to database: {e}")
    st.stop()

response_cache = get_response_cache()
query_logger = get_query_logger()
escalation_handler = get_escalation_handler()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("E.g., Who is the director?"):
    
    # Add user message to UI immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        start_time = time.time()
        cache_hit = False
        source_urls = []
        
        # 1. Check escalation first
        escalation_response = escalation_handler.check_escalation(prompt)
        
        if escalation_response:
            clean_answer = escalation_response
            message_placeholder.markdown(f"🚨 **Escalated Issue:**\n\n{clean_answer}")
            
        else:
            # 2. Check Cache
            cached_answer = response_cache.get(prompt)
            if cached_answer is not None:
                clean_answer = cached_answer
                cache_hit = True
                message_placeholder.markdown(f"⚡ **(Cached Response)**\n\n{clean_answer}")
                
            else:
                # 3. Query LLM & Vector DB
                with st.spinner("Searching the knowledge base..."):
                    try:
                        # Compile recent chat history (last 4 messages to save tokens/distraction)
                        chat_history_str = ""
                        for msg in st.session_state.messages[-4:]:
                            role = "User" if msg["role"] == "user" else "Assistant"
                            chat_history_str += f"{role}: {msg['content']}\n\n"

                        response = rag_chain.invoke({
                            "question": prompt,
                            "chat_history": chat_history_str
                        })
                        clean_answer = clean_llm_response(response)
                        
                        # Save to cache for next time
                        response_cache.put(prompt, clean_answer)
                        
                        # Display
                        message_placeholder.markdown(clean_answer)
                    except Exception as e:
                        clean_answer = f"Sorry, an error occurred while generating the answer: {e}"
                        message_placeholder.markdown(clean_answer)

        # Log query metrics for the professor
        response_time_ms = (time.time() - start_time) * 1000
        try:
            query_logger.log_query(
                question=prompt,
                answer=clean_answer,
                source_urls=source_urls,
                response_time_ms=response_time_ms,
                cache_hit=cache_hit,
                platform="streamlit_web"
            )
        except Exception:
            pass

    # Save assistant response in chat history
    st.session_state.messages.append({"role": "assistant", "content": clean_answer})
