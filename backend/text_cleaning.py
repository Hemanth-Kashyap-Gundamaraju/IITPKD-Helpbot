import re

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# A line longer than this is treated as "real content" (a sentence/title),
# not a navigation menu item. Menu items are usually short ("About Us"),
# while real titles (notifications, events) tend to run a bit longer.
MAX_NAV_LINE_LENGTH = 40

# If we see this many short, nav-looking lines in a row, we treat that whole
# block as a navigation/footer menu and remove it. Raised higher than before
# so smaller real content lists (e.g. 5-10 notifications) survive, and only
# genuinely huge menus (50+ lines) get cut.
MIN_NAV_CLUSTER_SIZE = 20

# Matches any digit. Real content (event years, notification titles, phone
# numbers, PIN codes) very often contains a number. Plain nav menu items
# ("Faculty", "Contact", "Research") almost never do.
DIGIT_PATTERN = re.compile(r"\d")


def _remove_duplicate_lines(text):
    """
    Description: Removes lines that repeat exactly within the same page.
        Websites often render the same navigation menu twice (mobile and
        desktop versions) - this keeps only the first copy of each line.
        Pure transformation, no side effects.
    Inputs: text (string). No globals read.
    Outputs: returns a string. No globals changed.
    Dependencies: none.
    Utilities: called by clean_page_text().
    """
    seen_lines = set()
    kept_lines = []

    for line in text.split("\n"):
        stripped_line = line.strip()
        if not stripped_line:
            kept_lines.append(line)
            continue
        if stripped_line not in seen_lines:
            seen_lines.add(stripped_line)
            kept_lines.append(line)

    return "\n".join(kept_lines)


def _is_probably_nav_line(line):
    """
    Description: Checks whether a single line looks like a navigation/menu
        item - short and with no digits in it. Real content (dates, years,
        phone numbers, PIN codes, notification titles) tends to break at
        least one of these rules.
    Inputs: line (string). Reads globals MAX_NAV_LINE_LENGTH, DIGIT_PATTERN.
    Outputs: returns True/False. No globals changed.
    Dependencies: none.
    Utilities: called by _remove_navigation_clusters().
    """
    stripped_line = line.strip()
    if not stripped_line or len(stripped_line) > MAX_NAV_LINE_LENGTH:
        return False
    if DIGIT_PATTERN.search(stripped_line):
        return False
    return True


def _remove_navigation_clusters(text):
    """
    Description: Deletes big blocks of short, menu-like lines (e.g. 'About
        Us', 'Faculty', 'Contact' one after another). Real page content is
        either longer or contains numbers, so this heuristic leaves that
        untouched.
    Inputs: text (string). Reads global MIN_NAV_CLUSTER_SIZE.
    Outputs: returns a string. No globals changed.
    Dependencies: calls _is_probably_nav_line().
    Utilities: called by clean_page_text().
    """
    lines = text.split("\n")
    kept_lines = []
    current_cluster = []

    def _flush_cluster():
        if len(current_cluster) >= MIN_NAV_CLUSTER_SIZE:
            return
        kept_lines.extend(current_cluster)

    for line in lines:
        if _is_probably_nav_line(line):
            current_cluster.append(line)
        else:
            _flush_cluster()
            current_cluster = []
            kept_lines.append(line)

    _flush_cluster()
    return "\n".join(kept_lines)


def clean_page_text(text):
    """
    Description: Runs one page's raw scraped text through both cleaning
        steps - duplicate-line removal, then navigation-cluster removal.
        Pure transformation, easy to unit test with plain strings.
    Inputs: text (string). No globals read directly (its helpers do).
    Outputs: returns a cleaned string. No globals changed.
    Dependencies: calls _remove_duplicate_lines(), _remove_navigation_clusters().
    Utilities: called by clean_scraped_documents().
    """
    text = _remove_duplicate_lines(text)
    return _remove_navigation_clusters(text)


def clean_scraped_documents(documents):
    """
    Description: Cleans up every scraped document in place, so real content
        (like an address or a notification) isn't drowned out when the
        text gets chunked later.
    Inputs: documents (list of Document objects). No globals read directly.
    Outputs: returns the same list, with each doc.page_content cleaned.
        Mutates each document's page_content attribute.
    Dependencies: calls clean_page_text().
    Utilities: called by backend.scraper.scrape_target_pages().
    """
    for doc in documents:
        doc.page_content = clean_page_text(doc.page_content)
    return documents