import re

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# Compiled once so we don't rebuild this regex every time we clean a response.
# Matches everything between <think> and </think>, including newlines (DOTALL).
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def extract_answer_text(response):
    """
    Pulls the plain text answer out of a LangChain LLM response.
    Some models return content as a list of blocks (e.g. [{'text': ...}]),
    others return it as a plain string. This handles both cases.
    """
    if isinstance(response.content, list):
        return response.content[0]["text"]
    return response.content


def strip_thinking_tags(text):
    """
    Removes any <think>...</think> block from the model's answer.
    Some models (like qwen3.6) show their internal reasoning in these tags
    before giving the real answer - we don't want to show that to users.
    """
    return THINK_TAG_PATTERN.sub("", text).strip()


def clean_llm_response(response):
    """
    One-stop helper: takes the raw LangChain response and returns the
    final, user-ready answer text (no thinking tags, no extra formatting).
    """
    raw_text = extract_answer_text(response)
    return strip_thinking_tags(raw_text)
