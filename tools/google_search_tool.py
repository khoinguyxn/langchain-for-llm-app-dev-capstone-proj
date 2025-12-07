"""Google Search tool with LLM-optimized output sanitization.

Wraps SerpAPI to provide clean, token-efficient search results for LangChain agents.
Prioritizes high-quality information sources and removes metadata/tracking data.

Environment:
    SERPAPI_API_KEY: SerpAPI authentication key (required)

Dependencies:
    python-dotenv>=1.2.1
"""

from langchain.tools import tool
from libs.serpapi.search import search, sanitize_search_results


@tool
def google_search_tool(query: str) -> str:
    """Execute Google search via SerpAPI and return LLM-optimized results.

    Sanitizes raw SerpAPI response into a structured format with hierarchical
    information priority: Knowledge Graph > Answer Box > Top 3 Organic Results.
    Output limited to ~200-500 tokens for efficient LLM processing.

    Args:
        query: Search query string.

    Returns:
        Formatted search results with sections for Knowledge Graph, Answer Box,
        and top 3 organic results (title, snippet, source URLs). Returns fallback
        message if no results found.

    Raises:
        Exception: Generic handler for API/network failures.

    Example:
        >>> result = google_search_tool.invoke({"query": "Python async programming"})
        >>> print(result)
        Knowledge Graph:
        **Asynchronous I/O** (Programming paradigm)
        ...
    """
    if not query or not query.strip():
        return "No results found for your query."

    # Perform search
    results = search(query, engine="google")

    # Check for errors
    if "error" in results:
        return f"Search failed: {results['error']}"

    # Sanitize and format results
    formatted_output, _ = sanitize_search_results(results)

    return formatted_output
