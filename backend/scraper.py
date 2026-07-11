import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
import config

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# A line longer than this is treated as "real content" (a sentence), not a
# navigation menu item. Menu items like "About Us" or "Contact" are short.
MAX_NAV_LINE_LENGTH = 60

# If we see this many short lines in a row, we treat that whole block as a
# navigation/footer menu and remove it, since real writing rarely has this
# many short lines back-to-back.
MIN_NAV_CLUSTER_SIZE = 8


def discover_internal_links(base_url=None, max_pages=None):
    """Scans the base URL and dynamically discovers internal links."""
    base_url = base_url or config.SCRAPER_BASE_URL
    max_pages = max_pages or config.SCRAPER_DISCOVERY_MAX_PAGES
    print(f"🔍 Dynamic Discovery: Scanning {base_url} for internal links...")

    discovered_urls = set([base_url])
    urls_to_crawl = [base_url]
    base_domain = urlparse(base_url).netloc

    try:
        while urls_to_crawl and len(discovered_urls) < max_pages:
            current_url = urls_to_crawl.pop(0)
            response = requests.get(
                current_url,
                headers=config.SCRAPER_HEADERS,
                timeout=config.SCRAPER_REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            for anchor in soup.find_all("a", href=True):
                href = anchor["href"]
                # Resolve relative paths to absolute URLs
                full_url = urljoin(base_url, href).split("#")[0].rstrip("/")

                # Check constraints:
                # 1. Must stay within the same domain (e.g., iitpkd.ac.in)
                # 2. Skip non-html file links (PDFs, images, zip files)
                if urlparse(full_url).netloc == base_domain:
                    if not full_url.lower().endswith(config.SCRAPER_SKIP_EXTENSIONS):
                        if (
                            full_url not in discovered_urls
                            and len(discovered_urls) < max_pages
                        ):
                            discovered_urls.add(full_url)
                            urls_to_crawl.append(full_url)

    except Exception as e:
        print(f"⚠️ Warning during link discovery: {e}")

    print(f"🎯 Found {len(discovered_urls)} unique internal links to ingest.")
    return list(discovered_urls)


def _remove_duplicate_lines(text):
    """
    Removes lines that repeat exactly within the same page.
    Websites often render the same navigation menu twice (once for mobile,
    once for desktop) - this keeps only the first copy of each line.
    """
    seen_lines = set()
    kept_lines = []

    for line in text.split("\n"):
        stripped_line = line.strip()

        # Always keep blank lines as-is, they don't hurt anything.
        if not stripped_line:
            kept_lines.append(line)
            continue

        if stripped_line not in seen_lines:
            seen_lines.add(stripped_line)
            kept_lines.append(line)

    return "\n".join(kept_lines)


def _is_probably_nav_line(line):
    """A short line is probably a navigation/menu item, not a real sentence."""
    stripped_line = line.strip()
    return 0 < len(stripped_line) <= MAX_NAV_LINE_LENGTH


def _remove_navigation_clusters(text):
    """
    Deletes big blocks of short, menu-like lines (e.g. 'About Us', 'Faculty',
    'Contact' one after another). Real page content (paragraphs, addresses,
    descriptions) is made of longer sentences, so this leaves that untouched.
    """
    lines = text.split("\n")
    kept_lines = []
    current_cluster = []

    def flush_cluster():
        """Decides whether the lines collected so far are a nav menu or real content."""
        if len(current_cluster) >= MIN_NAV_CLUSTER_SIZE:
            # Too many short lines in a row -> treat as navigation clutter, drop it.
            return
        kept_lines.extend(current_cluster)

    for line in lines:
        if _is_probably_nav_line(line):
            current_cluster.append(line)
        else:
            flush_cluster()
            current_cluster = []
            kept_lines.append(line)

    flush_cluster()  # handle any cluster left at the very end of the page
    return "\n".join(kept_lines)


def clean_scraped_documents(documents):
    """
    Cleans up every scraped document by stripping out duplicate lines and
    navigation menu clutter, so real content (like an address or a
    description) isn't drowned out when the text gets chunked later.
    """
    for doc in documents:
        cleaned_text = _remove_duplicate_lines(doc.page_content)
        cleaned_text = _remove_navigation_clusters(cleaned_text)
        doc.page_content = cleaned_text
    return documents


def scrape_target_pages():
    """Dynamically gathers links and returns loaded, cleaned document contexts."""
    # 1. Dynamically crawl and discover links from the landing page
    dynamic_urls = discover_internal_links(max_pages=config.SCRAPER_TARGET_MAX_PAGES)

    # 2. Feed the dynamic list directly into LangChain's loader
    loader = WebBaseLoader(web_paths=dynamic_urls)
    raw_documents = loader.load()

    # 3. Strip out repeated nav/footer clutter before this text gets chunked
    return clean_scraped_documents(raw_documents)