import sys
import time
import threading
from urllib import response
from backend.response_utils import clean_llm_response


def loading_counter(stop_event):
    """Prints a live ticking timer on the command line during network calls."""
    start_time = time.time()
    while not stop_event.is_set():
        elapsed = int(time.time() - start_time)
        sys.stdout.write(f"\rThinking... [{elapsed}s]")
        sys.stdout.flush()
        time.sleep(1)
    # Clear the loading line once finished
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()


def execute_query_with_loading(rag_chain, user_question):
    """Manages the UI loading thread context while executing the model invocation."""
    stop_loading = threading.Event()
    counter_thread = threading.Thread(target=loading_counter, args=(stop_loading,))

    try:
        counter_thread.start()

        # Invoke network request to Groq pipeline
        response = rag_chain.invoke(user_question)

        stop_loading.set()
        counter_thread.join()

        # Clean structural output strings out of Langchain components
        clean_answer = clean_llm_response(response)
        print(f"Answer: {clean_answer}")

    except Exception as e:
        stop_loading.set()
        if counter_thread.is_alive():
            counter_thread.join()
        print(f"\nAn execution error occurred: {e}")


def run_chat_loop(rag_chain):
    """Runs the terminal shell loop capturing user text input instructions."""
    print("\n--- Targeted Website Chatbot Initialized (Modular Architecture) ---")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_question = input("\nYour Question: ")
        if user_question.lower() in ["exit", "quit"]:
            print("Shutting down chatbot. Goodbye!")
            break

        if not user_question.strip():
            continue

        execute_query_with_loading(rag_chain, user_question)
