import sys
import time
import threading
from backend.utils.response_utils import clean_llm_response
from backend.utils.cache_utils import get_response_cache
from backend.utils.analytics import get_query_logger
from backend.utils.escalation import get_escalation_handler


class LoadingIndicator:
    """
    Description: Shows a live ticking "Thinking... [Ns]" counter on the
        terminal while a slow network call runs in the background. Groups
        together the thread and its stop-signal (the same two pieces of
        data that used to be passed around loose between functions) as
        attributes on one object.
    Inputs (constructor): none.
    Utilities: used by execute_query_with_loading().
    """

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._tick)

    def _tick(self):
        """
        Description: The background loop that actually prints the ticking
            counter, once per second, until told to stop.
        Inputs: none. Reads self._stop_event.
        Outputs: no return value. Writes to stdout as a side effect.
        Dependencies: none.
        Utilities: run as the target of self._thread (started by start()).
        """
        start_time = time.time()
        while not self._stop_event.is_set():
            elapsed = int(time.time() - start_time)
            sys.stdout.write(f"\rThinking... [{elapsed}s]")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\r" + " " * 30 + "\r")
        sys.stdout.flush()

    def start(self):
        """
        Description: Starts the background counter thread.
        Inputs: none.
        Outputs: no return value. Starts self._thread.
        Dependencies: none.
        Utilities: called by execute_query_with_loading().
        """
        self._thread.start()

    def stop(self):
        """
        Description: Signals the background thread to stop and waits for
            it to finish, so the ticking line is fully cleared before
            anything else prints.
        Inputs: none. Sets self._stop_event.
        Outputs: no return value.
        Dependencies: none.
        Utilities: called by execute_query_with_loading().
        """
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()


def execute_query_with_loading(rag_chain, user_question):
    """
    Description: Runs one question through the RAG chain while showing a
        loading indicator, then prints the cleaned answer. Handles errors
        so one bad query doesn't crash the whole chat loop.
    Inputs: rag_chain (the LangChain runnable), user_question (string).
        No globals read.
    Outputs: no return value. Prints the answer (or an error) to stdout.
    Dependencies: calls LoadingIndicator, clean_llm_response(); calls
        rag_chain.invoke().
    Utilities: called by run_chat_loop().
    """
    indicator = LoadingIndicator()
    response_cache = get_response_cache()
    query_logger = get_query_logger()
    escalation_handler = get_escalation_handler()

    start_time = time.time()
    source_urls = []
    cache_hit = False

    try:
        escalation_response = escalation_handler.check_escalation(user_question)
        
        if escalation_response:
            print(f"Answer (Escalated): {escalation_response}")
            clean_answer = escalation_response
        else:
            cached_answer = response_cache.get(user_question)
            if cached_answer is not None:
                print(f"Answer (Cache Hit): {cached_answer}")
                clean_answer = cached_answer
                cache_hit = True
            else:
                indicator.start()
                response = rag_chain.invoke(user_question)
                indicator.stop()

                clean_answer = clean_llm_response(response)
                response_cache.put(user_question, clean_answer)
                print(f"Answer: {clean_answer}")

        response_time_ms = (time.time() - start_time) * 1000

        # Log the interaction
        query_logger.log_query(
            question=user_question,
            answer=clean_answer,
            source_urls=source_urls,
            response_time_ms=response_time_ms,
            cache_hit=cache_hit,
            platform="cli"
        )

    except Exception as e:
        indicator.stop()
        print(f"\nAn execution error occurred: {e}")


def run_chat_loop(rag_chain):
    """
    Description: Runs the terminal shell loop that reads user questions and
        answers them, until the user types 'exit' or 'quit'.
    Inputs: rag_chain (the LangChain runnable). No globals read.
    Outputs: no return value. Runs until the user exits.
    Dependencies: calls execute_query_with_loading().
    Utilities: called by main.py.
    """
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