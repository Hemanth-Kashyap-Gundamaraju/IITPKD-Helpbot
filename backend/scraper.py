from langchain_community.document_loaders import WebBaseLoader
import config
from backend.link_discovery import LinkDiscoverer
from backend.text_cleaning import clean_scraped_documents


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
        main.py) and cron_ingest.run_ingestion().
    """
    discoverer = LinkDiscoverer(max_pages=config.SCRAPER_TARGET_MAX_PAGES)
    dynamic_urls = discoverer.crawl()

    loader = WebBaseLoader(web_paths=dynamic_urls)
    raw_documents = loader.load()

    return clean_scraped_documents(raw_documents)