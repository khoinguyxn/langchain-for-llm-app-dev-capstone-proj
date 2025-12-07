"""
Chroma vector database client setup with Ollama embeddings.

Centralizes Chroma client initialization to avoid code duplication.
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import Optional

# Singleton Chroma client for vector DB operations
_chroma_client: Optional[Chroma] = None


def get_embeddings():
    """Retrieve embedding function for Chroma client.

    Returns:
        Embedding function instance.
    """
    return OllamaEmbeddings(model="nomic-embed-text:latest")


def get_chroma_client() -> Chroma:
    """Get or create singleton Chroma client instance.

    Lazy-loads the client on first access to ensure efficient resource usage.

    Returns:
        Configured Chroma client instance.
    """
    global _chroma_client

    if _chroma_client is None:
        embeddings = get_embeddings()

        _chroma_client = Chroma(
            collection_name="my_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
        )

    return _chroma_client
