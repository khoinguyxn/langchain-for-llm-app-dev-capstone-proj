"""Text splitters for web content loaded via WebBaseLoader.

Designed to work with WebBaseLoader which extracts text content from HTML pages.
Uses intelligent text-based splitting with semantic separators.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


def split_text_content(
    docs: List[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split web documents using semantic text-based chunking.

    Optimized for WebBaseLoader output. Uses hierarchical separators to
    preserve semantic structure (paragraphs, sentences, words) while
    maintaining chunk size limits.

    Args:
        docs: List of Document objects from WebBaseLoader.
        chunk_size: Maximum characters per chunk (default: 1500 for Wikipedia).
        chunk_overlap: Overlap between chunks for context continuity (default: 200).

    Returns:
        List of split Document objects with preserved metadata (source URL, etc).

    Example:
        >>> from langchain_community.document_loaders import WebBaseLoader
        >>> loader = WebBaseLoader("https://en.wikipedia.org/wiki/Python")
        >>> docs = loader.load()
        >>> chunks = split_text_content(docs)
        >>> print(f"Split {len(docs)} docs into {len(chunks)} chunks")
        >>> print(chunks[0].metadata)  # Preserves source URL
    """
    # Use RecursiveCharacterTextSplitter with semantic separators
    # Priority: double newline (paragraphs) > newline > sentence > word > char
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",  # Paragraph breaks (highest priority)
            "\n",    # Line breaks
            ". ",    # Sentence endings
            " ",     # Word boundaries
            "",      # Character level (fallback)
        ],
    )

    # Split documents while preserving all metadata
    chunks = text_splitter.split_documents(docs)

    return chunks
