"""Module for performing searches using SerpAPI.
Provides functions for searching and sanitizing results from various engines
like Google and Google Scholar with consistent error handling.

Dependencies:
    serpapi>=0.1.5
"""

from .config import get_serpapi_client
from typing import Dict, Any, List, Tuple


def sanitize_search_results(results: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Sanitize and format SerpAPI search results for LLM consumption.

    Extracts and formats results with hierarchical priority:
    1. Knowledge Graph (highest quality)
    2. Answer Box (direct answers)
    3. Top 3 Organic Results (with source URLs)

    Args:
        results: Raw SerpAPI response dictionary.

    Returns:
        Tuple of (formatted_string, list_of_source_urls).
        - formatted_string: Human-readable search results
        - list_of_source_urls: URLs for potential deep research

    Example:
        >>> results = search("Python programming")
        >>> formatted, urls = sanitize_search_results(results)
        >>> print(formatted)  # Shows Knowledge Graph + snippets
        >>> print(urls)  # ['https://python.org', 'https://...']
    """
    sanitized_results = []
    source_urls = []

    # 1. Extract Knowledge Graph (highest quality, direct answer)
    if "knowledge_graph" in results and isinstance(results["knowledge_graph"], dict):
        kg = results["knowledge_graph"]
        title = str(kg.get("title", "N/A"))[:200]
        kg_info = f"**{title}**"

        if "type" in kg and isinstance(kg["type"], str):
            kg_info += f" ({str(kg['type'])[:100]})"

        if "description" in kg and isinstance(kg["description"], str):
            kg_info += f"\n{str(kg['description'])[:500]}"

        sanitized_results.append(f"Knowledge Graph:\n{kg_info}")

        # Extract Knowledge Graph source URL
        if "source" in kg and isinstance(kg["source"], dict):
            kg_url = kg["source"].get("link")
            if kg_url:
                source_urls.append(str(kg_url))

    # 2. Extract Answer Box (direct answer from Google)
    if "answer_box" in results and isinstance(results["answer_box"], dict):
        answer = results["answer_box"]
        answer_text = answer.get("answer") or answer.get("snippet") or ""

        if answer_text and isinstance(answer_text, str):
            sanitized_results.append(f"Direct Answer:\n{str(answer_text)[:300]}")

    # 3. Extract Organic Results (top 3 results)
    if "organic_results" in results and isinstance(results["organic_results"], list):
        top3_or = results["organic_results"][:3]

        for i, result in enumerate(top3_or, 1):
            if not isinstance(result, dict):
                continue

            title = str(result.get("title", "No title"))[:150]
            snippet = str(result.get("snippet", "No snippet"))[:300]
            link = str(result.get("link", ""))[:200]

            if link:
                source_urls.append(link)

            sanitized_results.append(
                f"Result {i}:\nTitle: {title}\nSnippet: {snippet}\nSource: {link}"
            )

    # Combine all results
    formatted_output = "\n\n".join(sanitized_results) if sanitized_results else "No results found for your query."

    return formatted_output, source_urls


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
