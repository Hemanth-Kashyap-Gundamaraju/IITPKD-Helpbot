from fastapi import FastAPI
from pathlib import Path
from dotenv import load_dotenv
import backend.chain_builder as chain_builder
import interface.cli as cli

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
app = FastAPI()
# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# Pass "--refresh" as a command line flag when you actually want to
# re-scrape the website and re-embed everything from scratch.
# Example: python main.py --refresh
FORCE_REFRESH_FLAG = "--refresh"


def _build_chain():
    """
    Description: Loads the existing Pinecone-backed vector store without
        re-embedding anything.
    Inputs: none. No globals read directly.
    Outputs: returns a RAG chain (LangChain runnable). No globals changed.
    Dependencies: calls backend.chain_builder.build_rag_chain() with no
        retriever so it can connect to Pinecone.
    Utilities: called by main().
    """
    return chain_builder.build_rag_chain(retriever=None)


def main():
    """
    Description: Main orchestration entry point that builds the RAG chain
        from the configured vector store and hands off to the CLI chat loop.
    Inputs: none. No globals read directly (its dependencies do).
    Outputs: no return value.
    Dependencies: calls _build_chain(), interface.cli.run_chat_loop().
    Utilities: called by the `if __name__ == "__main__"` block below (i.e.
        this is the script's own entry point when run directly).
    """

    rag_chain = _build_chain()

    cli.run_chat_loop(rag_chain)


if __name__ == "__main__":
    main()
