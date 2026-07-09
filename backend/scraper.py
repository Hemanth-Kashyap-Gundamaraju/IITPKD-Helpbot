import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
import config

def safe_html_extractor(html: str) -> str:
    """Cleans up raw HTML files, stripping styling, headers, and navigation scripts."""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "header", "footer", "nav"]):
        element.decompose()
    return re.sub(r"\n+", "\n", soup.get_text()).strip()

def scrape_target_pages():
    """Scrapes institutional pages and returns processed clean documents."""
    print("\nScraping institutional pages...")
    loader = WebBaseLoader(config.TARGET_URLS)
    documents = loader.load()

    for doc in documents:
        doc.page_content = safe_html_extractor(doc.page_content)

    print(f"Successfully scraped {len(documents)} pages.")
    return documents