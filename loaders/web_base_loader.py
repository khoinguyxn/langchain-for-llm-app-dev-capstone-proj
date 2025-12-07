"""Web page loader with consistent error handling.

Provides async loading of web pages with proper type-safe error handling.
"""

from langchain_community.document_loaders import WebBaseLoader
from typing import List
from langchain_core.documents import Document
from typing import Optional


async def load_web_page(url: str) -> Optional[List[Document]]:
    """Load a web page asynchronously.

    Args:
        url: URL to load.

    Returns:
        List of Document objects if successful, None if loading failed.
    """
    try:
        loader = WebBaseLoader(url)
        pages: List[Document] = []

        async for document in loader.alazy_load():
            pages.append(document)

        return pages
    except Exception as e:
        print(f"Failed to load web page {url}: {str(e)[:100]}")

        return None
