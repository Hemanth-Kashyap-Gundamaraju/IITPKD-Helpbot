"""
Description: Unit tests for backend.utils.response_utils. We don't have a
    real LangChain LLM response object here (that would need a real network
    call to Groq), so we build a tiny fake stand-in class with just a
    `.content` attribute - that's all these functions actually read.
"""
from backend.utils.response_utils import (
    _extract_answer_text,
    _strip_thinking_tags,
    clean_llm_response,
)


class FakeResponse:
    """
    Description: A cheap stand-in for a real LangChain LLM response object.
        Real responses have many attributes, but our functions only ever
        read `.content`, so that's all we bother faking.
    Inputs (constructor): content (string OR list - matches the two shapes
        clean_llm_response() needs to handle).
    Utilities: used by tests in this file only.
    """

    def __init__(self, content):
        self.content = content


def test_extract_answer_text_with_plain_string():
    """Some models return response.content as a plain string."""
    response = FakeResponse("Hello world")
    assert _extract_answer_text(response) == "Hello world"


def test_extract_answer_text_with_list_of_blocks():
    """Other models return response.content as a list of {'text': ...} blocks."""
    response = FakeResponse([{"text": "Hello world"}])
    assert _extract_answer_text(response) == "Hello world"


def test_strip_thinking_tags_removes_think_block():
    """A <think>...</think> block anywhere in the text should be removed."""
    raw_text = "<think>internal reasoning here</think>The real answer."
    assert _strip_thinking_tags(raw_text) == "The real answer."


def test_strip_thinking_tags_leaves_normal_text_unchanged():
    """If there's no <think> tag at all, text should pass through untouched."""
    raw_text = "Just a normal answer with no thinking tags."
    assert _strip_thinking_tags(raw_text) == raw_text


def test_strip_thinking_tags_handles_multiline_think_block():
    """<think> blocks can span multiple lines - DOTALL should catch this."""
    raw_text = "<think>\nline one\nline two\n</think>Final answer here."
    assert _strip_thinking_tags(raw_text) == "Final answer here."


def test_clean_llm_response_end_to_end_with_think_tag():
    """Full pipeline: extract text from a fake response AND strip its think tag."""
    response = FakeResponse("<think>reasoning</think>Clean final answer")
    assert clean_llm_response(response) == "Clean final answer"
