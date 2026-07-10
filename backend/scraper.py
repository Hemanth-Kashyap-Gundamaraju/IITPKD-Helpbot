import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
import config

def safe_html_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    
    for element in soup(["script", "style"]):
        element.decompose()
        
    text_blocks = []
    for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'td', 'li', 'span']):
        text = tag.get_text().strip()
        if text:
            text_blocks.append(text)
            
    clean_text = "\n".join(text_blocks)
    
    # EMERGENCY FALLBACK: If explicit tags returned nothing due to a structural block or error page,
    # capture the raw text content so the chunk list doesn't drop to 0.
    if not clean_text.strip():
        clean_text = soup.get_text()

    return re.sub(r"\n+", "\n", clean_text).strip()
def scrape_target_pages():
    """Scrapes institutional pages and returns processed clean documents."""
    print("\nScraping institutional pages...")
    loader = WebBaseLoader(config.TARGET_URLS)
    documents = loader.load()

    for doc in documents:
        doc.page_content = safe_html_extractor(doc.page_content)

    print(f"Successfully scraped {len(documents)} pages.")
    return documents