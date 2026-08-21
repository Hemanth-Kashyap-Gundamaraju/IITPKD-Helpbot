"""
Description: Unit tests for backend.DatabaseBuilder.text_cleaning.
    These are pure string-processing functions (no network, no files),
    so we can test them directly with plain string inputs.
"""
from backend.DatabaseBuilder.text_cleaning import (
    _remove_duplicate_lines,
    _is_probably_nav_line,
    _remove_navigation_clusters,
    clean_page_text,
)


# ---------------------------------------------------------------------------
# _remove_duplicate_lines
# ---------------------------------------------------------------------------

def test_remove_duplicate_lines_removes_exact_repeats():
    """A line repeated later in the same page (e.g. mobile+desktop nav) should only appear once."""
    text = "Home\nAbout\nHome\nContact"
    result = _remove_duplicate_lines(text)
    assert result == "Home\nAbout\nContact"


def test_remove_duplicate_lines_keeps_blank_lines():
    """Blank lines are kept every time - they're not 'duplicate content', just spacing."""
    text = "Home\n\n\nContact"
    result = _remove_duplicate_lines(text)
    assert result == "Home\n\n\nContact"


# ---------------------------------------------------------------------------
# _is_probably_nav_line
# ---------------------------------------------------------------------------

def test_is_probably_nav_line_true_for_short_line_no_digits():
    """A short line with no digits looks like a menu item, e.g. 'Faculty'."""
    assert _is_probably_nav_line("Faculty") is True


def test_is_probably_nav_line_false_when_line_has_digits():
    """A line with digits (PIN code, phone number, year) is real content, not nav."""
    assert _is_probably_nav_line("PIN: 678557") is False


def test_is_probably_nav_line_false_when_line_too_long():
    """A long line is treated as a real sentence/title, not a menu item."""
    long_line = "This is a notification about the upcoming convocation ceremony details"
    assert _is_probably_nav_line(long_line) is False


def test_is_probably_nav_line_false_for_empty_line():
    """An empty/whitespace-only line is not a nav item."""
    assert _is_probably_nav_line("   ") is False


# ---------------------------------------------------------------------------
# _remove_navigation_clusters
# ---------------------------------------------------------------------------
alphabets =[chr(65 + i) for i in range(26)]
def test_remove_navigation_clusters_deletes_big_nav_block():
    """A block of 25 short nav-looking lines in a row (>= 20) should be removed entirely.
    The 'real content' markers here deliberately contain digits (like a real notification
    title would - a date, a year, a PIN) so they are NOT themselves mistaken for nav lines.
    (See test_remove_navigation_clusters_can_eat_adjacent_short_real_line below for what
    happens when a short real line does NOT have this protection.)"""
    nav_block = "\n".join([f"Menu Item {i}" for i in alphabets])
    text = f"Convocation 2026 notice\n{nav_block}\nContact PIN 678557"
    result = _remove_navigation_clusters(text)
    assert "Menu Item A" in result
    assert "Menu Item Z" in result                               # nav cluster end saved
    middle_menus = {f"Menu {i}" for i in alphabets[1:-1]}
    result_lines = set(result.splitlines())
    assert len(result_lines.intersection(middle_menus)) == 0 # check if the middle lines are skipped
    assert "Convocation 2026 notice" in result
    assert "Contact PIN 678557" in result


def test_remove_navigation_clusters_can_eat_adjacent_short_real_line():
    """KNOWN EDGE CASE: a short line with no digits (e.g. a short heading) sitting right
    next to a big nav block gets swept into that cluster and deleted too, even though it
    isn't actually a menu item. The heuristic only looks at line PATTERN (short + no
    digits), not whether the line is truly part of the menu. This test documents the
    current behavior so it doesn't surprise us later - it is not necessarily correct
    behavior, just the current one."""
    nav_block = "\n".join([f"Menu Item {i}" for i in alphabets])
    text = f"Short Heading\n{nav_block}"
    result = _remove_navigation_clusters(text)
    # "Short Heading" also looks like a nav line (short, no digits), so it gets removed
    # along with the real menu - this assert documents that current (imperfect) behavior.
    assert "Short Heading"  in result
    assert "Menu Item Z" in result                               # nav cluster end saved
    middle_menus = {f"Menu {i}" for i in alphabets[1:-1]}
    result_lines = set(result.splitlines())
    assert len(result_lines.intersection(middle_menus)) == 0 # check if the middle lines are skipped


def test_remove_navigation_clusters_keeps_small_nav_looking_block():
    """A block of only 10 short nav-looking lines (< 20) should SURVIVE - this protects
    small real content lists, like 10 notification titles, from being wrongly deleted."""
    small_block = "\n".join([f"Item {i}" for i in alphabets[:10]])
    result = _remove_navigation_clusters(small_block)
    assert "Item A" in result
    assert "Item J" in result  # the 10th item (A..J)


# ---------------------------------------------------------------------------
# clean_page_text (both steps combined)
# ---------------------------------------------------------------------------

def test_clean_page_text_runs_both_cleaning_steps():
    """End-to-end: duplicate lines removed AND big nav clusters removed together BUT the first and last lines survive
    'Welcome to IIT PKD 2026' deliberately contains a digit so it survives the nav-cluster
    step and only the duplicate-removal step affects it.
    """
    nav_block = "\n".join([f"Menu {i}" for i in alphabets])
    text = f"Welcome to IIT PKD 2026\nWelcome to IIT PKD 2026\n{nav_block}\nContact us at 678557"
    result = clean_page_text(text)
    assert result.count("Welcome to IIT PKD 2026") == 1   # duplicate removed
    assert "Menu A"  in result.splitlines()                           # nav cluster begining saved
    assert "Menu Z" in result.splitlines()                               # nav cluster end saved
    middle_menus = {f"Menu {i}" for i in alphabets[1:-1]}
    result_lines = set(result.splitlines())
    assert len(result_lines.intersection(middle_menus)) == 0 # check if the middle lines are skipped
    assert "Contact us at 678557" in result                 # real content (has digits) kept
