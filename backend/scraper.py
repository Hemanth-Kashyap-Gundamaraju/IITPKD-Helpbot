import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
import config


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


def scrape_target_pages():
    """Dynamically gathers links and returns loaded document contexts."""
    # 1. Dynamically crawl and discover links from the landing page
    dynamic_urls = discover_internal_links(max_pages=config.SCRAPER_TARGET_MAX_PAGES)

    # 2. Feed the dynamic list directly into LangChain's loader
    loader = WebBaseLoader(web_paths=dynamic_urls)
    return loader.load()
