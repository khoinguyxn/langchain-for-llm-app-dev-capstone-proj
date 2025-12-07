"""Module for performing searches using SerpAPI.

Provides a function to search various engines like Google and Google Scholar
with consistent error handling.

Dependencies:
    serpapi>=0.1.5
"""

from .config import get_serpapi_client
from typing import Dict, Any


def search(query: str, engine: str = "google") -> Dict[str, Any]:
    """Perform a search using SerpAPI with consistent error handling.

    Args:
        query: The search query string.
        engine: The search engine to use (default: "google").
                Options: "google", "google_scholar"

    Returns:
        Dictionary containing search results or error information.
        On error, returns dict with "error" key.
        Callers should check for "error" key to detect failures.

    Example:
        >>> results = search("Python programming")
        >>> if "error" in results:
        ...     print("Search failed")
        >>> else:
        ...     print(results.get("organic_results", []))
    """
    if not query or not query.strip():
        return {"error": "No results found for your query."}

    try:
        params = {
            "engine": engine,
            "q": query.strip()[:500],  # Limit query length
        }

        client = get_serpapi_client()
        results = client.search(params)

        # Convert to dict for consistent typing
        return dict(results) if results else {"error": "Empty response from API"}
    except Exception as e:
        print(f"SerpAPI search error: {e}")
        return {"error": f"Search failed: {str(e)[:100]}"}
