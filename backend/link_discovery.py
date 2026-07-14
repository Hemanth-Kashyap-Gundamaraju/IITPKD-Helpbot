import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import config


class LinkDiscoverer:
    """
    Description: Crawls a website starting from a base URL and collects
        internal links, staying on the same domain and skipping
        non-HTML files. Groups together the running state of a crawl
        (which URLs are already found, which are still queued, the
        domain we're restricted to) as attributes on one object instead
        of passing them separately between functions.
    Inputs (constructor): base_url (optional, defaults to
        config.SCRAPER_BASE_URL), max_pages (optional, defaults to
        config.SCRAPER_DISCOVERY_MAX_PAGES).
    Utilities: used by backend.scraper.scrape_target_pages().
    """

    def __init__(self, base_url=None, max_pages=None):
        self.base_url = base_url or config.SCRAPER_BASE_URL
        self.max_pages = max_pages or config.SCRAPER_DISCOVERY_MAX_PAGES
        self.base_domain = urlparse(self.base_url).netloc
        self.discovered_urls = {self.base_url}
        self.urls_to_crawl = [self.base_url]

    def _is_worth_crawling(self, full_url):
        """
        Description: Checks whether a discovered URL should be added to the
            crawl queue - same domain, not already seen, not a skippable
            file type (PDF, image, etc.), and we haven't hit max_pages yet.
        Inputs: full_url (string). Reads self.base_domain, self.discovered_urls,
            self.max_pages. Reads global config.SCRAPER_SKIP_EXTENSIONS.
        Outputs: returns True/False. No state changed.
        Dependencies: none.
        Utilities: called by _extract_links_from_page().
        """
        if urlparse(full_url).netloc != self.base_domain:
            return False
        if full_url.lower().endswith(config.SCRAPER_SKIP_EXTENSIONS):
            return False
        if full_url in self.discovered_urls:
            return False
        return len(self.discovered_urls) < self.max_pages

    def _extract_links_from_page(self, page_url):
        """
        Description: Fetches one page and queues up any new internal links
            found on it. This is the only method that makes a network
            request, so it's the only piece that needs a real (or mocked)
            connection to test.
        Inputs: page_url (string). Reads global config.SCRAPER_HEADERS,
            config.SCRAPER_REQUEST_TIMEOUT.
        Outputs: no return value. Adds new URLs to self.discovered_urls
            and self.urls_to_crawl.
        Dependencies: calls _is_worth_crawling(); uses requests.get(),
            bs4.BeautifulSoup.
        Utilities: called by crawl().
        """
        response = requests.get(
            page_url,
            headers=config.SCRAPER_HEADERS,
            timeout=config.SCRAPER_REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return

        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.find_all("a", href=True):
            full_url = urljoin(self.base_url, anchor["href"]).split("#")[0].rstrip("/")
            if self._is_worth_crawling(full_url):
                self.discovered_urls.add(full_url)
                self.urls_to_crawl.append(full_url)

    def crawl(self):
        """
        Description: Runs the full crawl: keeps visiting queued pages and
            collecting new links until either the queue empties or
            max_pages is reached. Any network error stops the crawl early
            but still returns whatever was found so far.
        Inputs: none. Reads self.urls_to_crawl, self.discovered_urls,
            self.max_pages.
        Outputs: returns a list of discovered URLs. No globals changed.
        Dependencies: calls _extract_links_from_page().
        Utilities: called by backend.scraper.scrape_target_pages().
        """
        print(f"Dynamic Discovery: Scanning {self.base_url} for internal links...")
        try:
            while self.urls_to_crawl and len(self.discovered_urls) < self.max_pages:
                current_url = self.urls_to_crawl.pop(0)
                self._extract_links_from_page(current_url)
        except Exception as e:
            print(f"Warning during link discovery: {e}")

        print(f"Found {len(self.discovered_urls)} unique internal links to ingest.")
        return list(self.discovered_urls)