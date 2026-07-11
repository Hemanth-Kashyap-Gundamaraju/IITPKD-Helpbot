from dotenv import load_dotenv
import backend.scraper as scraper
import backend.vector_store as vector_store
import backend.chain_builder as chain_builder
import cli as cli

load_dotenv()

def main():
    """Main orchestration pipeline workflow entrypoint."""
    # 1. Gather text documents via web scraper
    raw_docs = scraper.scrape_target_pages()
    
    # 2. Extract database search retriever index
    retriever = vector_store.initialize_vector_db(raw_docs)
    
    # 3. Assemble the model pipeline execution components
    rag_chain = chain_builder.build_rag_chain(retriever)
    
    # 4. Fire up the chatbot session loop
    cli.run_chat_loop(rag_chain)

if __name__ == "__main__":
    main()