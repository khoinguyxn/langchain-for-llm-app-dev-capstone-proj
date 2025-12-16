"""
Chroma vector database client setup with Ollama embeddings.

Centralizes Chroma client initialization to avoid code duplication.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def reset_chroma_client():
    """Reset the singleton Chroma client (useful for testing/debugging)."""
    global _chroma_client
    _chroma_client = None


def get_embeddings():
    """Retrieve embedding function for Chroma client.

    Returns:
        Embedding function instance.
    """
    return OllamaEmbeddings(
        model="nomic-embed-text",
    )


def create_chroma_client() -> Chroma:
    """Create a new Chroma client instance (non-singleton).

    Use this instead of get_chroma_client() when you need a fresh instance
    with updated configuration.

    Returns:
        New Chroma client instance.
    """
    embeddings = get_embeddings()

    return Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
