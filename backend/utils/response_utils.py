import re

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# Compiled once so we don't rebuild this regex every time we clean a response.
# Matches everything between <think> and </think>, including newlines (DOTALL).
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _extract_answer_text(response):
    """
    Description: Pulls the plain text answer out of a LangChain LLM
        response. Some models return content as a list of blocks
        (e.g. [{'text': ...}]), others return it as a plain string -
        this handles both cases. Marked private (leading underscore)
        since only clean_llm_response() in this file calls it.
    Inputs: response (a LangChain LLM response object). No globals read.
    Outputs: returns a string. No globals changed.
    Dependencies: none.
    Utilities: called by clean_llm_response().
    """
    if isinstance(response.content, list):
        return response.content[0]["text"]
    return response.content


def _strip_thinking_tags(text):
    """
    Description: Removes any <think>...</think> block from the model's
        answer. Handles well-formed blocks and blocks that were cut off
        due to max_tokens.
    """
    # 1. Remove well-formed <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2. If the model was cut off and never closed the tag, strip everything from <think> onwards
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    
    # If the text is completely empty after stripping, return a fallback message
    text = text.strip()
    if not text:
        return "I apologize, but my response was cut off before I could finish my thought. Please try asking your question again!"
        
    return text


def clean_llm_response(response):
    """
    Description: One-stop helper - takes the raw LangChain response and
        returns the final, user-ready answer text (no thinking tags, no
        extra formatting). This is the only function in this file meant
        to be called from outside it.
    Inputs: response (a LangChain LLM response object). No globals read
        directly (its helpers do).
    Outputs: returns a string. No globals changed.
    Dependencies: calls _extract_answer_text(), _strip_thinking_tags().
    Utilities: called by interface.cli.execute_query_with_loading() and
        wsgi.whatsapp_webhook().
    """
    raw_text = _extract_answer_text(response)
    return _strip_thinking_tags(raw_text)