"""Module for performing searches using SerpAPI.

Provides a function to search various engines like Google and Google Scholar.

Dependencies:
    serpapi>=0.1.5
"""

from .config import get_serpapi_client
from serpapi import SerpResults
from typing import Any


def search(query: str, engine: str = "google") -> SerpResults | Any:
    """
    Perform a Google search using SerpAPI and return sanitized results.

    Args:
        query (str): The search query.
        engine (str): The search engine to use. Default is "google".

    Returns:
        SerpResults: The raw results from SerpAPI.
    """
    if not query or not query.strip():
        return {"error": "No results found for your query."}

    try:
        params = {
            "engine": engine,
            "q": query,
        }

        client = get_serpapi_client()
        results = client.search(params)

        return results
    except Exception as e:
        print(f"SerpAPI search error: {e}")
        return {"error": "Search failed: Unexpected error occurred."}
