import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
import config

def safe_html_extractor(html: str) -> str:
    """Extracts valid body and text blocks from pages without destroying actual content chunks."""
    soup = BeautifulSoup(html, "html.parser")
    
    # 1. Remove ONLY non-text operational code
    for element in soup(["script", "style"]):
        element.decompose()
        
    # 2. Extract text selectively from semantic structural tags only
    # This prevents header/nav wrappers from deleting deep page text paragraphs
    text_blocks = []
    for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'td', 'li', 'span']):
        text = tag.get_text().strip()
        if text:
            text_blocks.append(text)
            
    # Join everything with space or clean newlines
    clean_text = "\n".join(text_blocks)
    
    # Clean up double spacing and formatting noise
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