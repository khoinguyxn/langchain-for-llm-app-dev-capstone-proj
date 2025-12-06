"""Google Scholar search tool for academic research with LLM-optimized output.

Provides sanitized academic search results from Google Scholar, optimized for
LLM consumption by extracting titles, snippets, and related searches.

Environment:
    SERPAPI_API_KEY: SerpAPI authentication key (required)

Dependencies:
    serpapi>=0.1.5, python-dotenv>=1.2.1
"""

from langchain.tools import tool
import serpapi
import os
from dotenv import load_dotenv

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY environment variable is required but not set")

serpapi_client = serpapi.Client(api_key=SERPAPI_API_KEY)


@tool
def google_scholar_tool(query: str) -> str:
    """Search Google Scholar for academic papers, research articles, and citations.

    Returns top 3 organic results and related searches with sanitized content.
    Output truncated to prevent token bloat (title:150, snippet:300).

    Args:
        query: Academic search query (e.g., "machine learning transformers", "climate change impact").

    Returns:
        Formatted string with organic results (title, snippet, source) and related searches.
        Returns fallback message if no results found.

    Raises:
        ValueError: If SERPAPI_API_KEY not set.
        Exception: Generic handler for API/network failures.

    Example:
        >>> result = google_scholar_tool.invoke({"query": "transformer architecture NLP"})
        >>> print(result)
        Result 1:
        Title: Attention Is All You Need
        Snippet: The dominant sequence transduction models...
        Source: https://arxiv.org/abs/1706.03762
    """
    if not query or not query.strip():
        return "No results found for your query."

    query = query.strip()[:500]

    try:
        params = {
            "engine": "google_scholar",
            "q": query,
        }

        results = serpapi_client.search(params)
    except Exception as e:
        print(f"SerpAPI search error: {e}")

        return f"Search failed: Unexpected error occurred."

    sanitized_results = []

    # 1. Extract top 3 organic results (papers/articles)
    if "organic_results" in results and isinstance(results["organic_results"], list):
        top3_or = results["organic_results"][:3]

        for i, result in enumerate(top3_or, 1):
            if not isinstance(result, dict):
                continue

            title = str(result.get("title", "No title"))[:150]
            snippet = str(result.get("snippet", "No snippet"))[:300]
            link = str(result.get("link", ""))[:200]

            sanitized_results.append(
                f"Result {i}:\nTitle: {title}\nSnippet: {snippet}\nSource: {link}"
            )

    # 2. Extract top 3 related searches for additional context
    if "related_searches" in results and isinstance(results["related_searches"], list):
        top3_rs = results["related_searches"][:3]

        for i, search in enumerate(top3_rs, 1):
            if not isinstance(search, dict):
                continue

            query = str(search.get("query", "No query"))[:150]
            link = str(result.get("link", ""))[:200]

            sanitized_results.append(f"Search {i}:\nQuery: {query}\nLink: {link}")

    # 3. Combine all results
    if sanitized_results:
        return "\n\n".join(sanitized_results)

    return "No results found for your query."
