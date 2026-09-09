from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()
import config
from backend.utils.link_discovery import LinkDiscoverer
from backend.DatabaseBuilder.text_cleaning import clean_scraped_documents


def scrape_target_pages():
    """
    Description: The main scraping entry point - discovers internal links
        starting from the college homepage, loads their page content, and
        cleans out navigation/menu clutter. Wires together LinkDiscoverer
        and text_cleaning rather than doing any of that work itself.
    Inputs: none. Reads global config.SCRAPER_TARGET_MAX_PAGES.
    Outputs: returns a list of cleaned Document objects. No globals changed.
    Dependencies: uses backend.link_discovery.LinkDiscoverer,
        langchain_community.document_loaders.WebBaseLoader,
        backend.text_cleaning.clean_scraped_documents().
    Utilities: called by backend.vector_store.initialize_vector_db() (via
        main.py).
    """
    discoverer = LinkDiscoverer(max_pages=config.SCRAPER_TARGET_MAX_PAGES)
    dynamic_urls = discoverer.crawl()

    raw_documents = []
    for url in dynamic_urls:
        try:
            loader = WebBaseLoader(web_paths=[url])
            raw_documents.extend(loader.load())
        except Exception as e:
            print(f"Warning: Failed to load {url} - {e}")
            
    print(f"Scraped {len(raw_documents)} documents from {len(dynamic_urls)} unique internal links.")
    return clean_scraped_documents(raw_documents)