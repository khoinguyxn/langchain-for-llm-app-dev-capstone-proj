from langchain.tools import tool
from libs.serpapi.search import search, sanitize_search_results
from langchain_community.document_loaders import WebBaseLoader


@tool
def google_deep_search_tool(query: str) -> str:
    """Perform deep research by searching and loading full page content.

    Combines quick search snippets with full page content loading.
    Returns both sanitized search results AND full content from top source.

    Use this when you need detailed, in-depth content beyond snippets.

    Args:
        query: Search query string.

    Returns:
        Formatted string with search snippets + full page content from top source.

    Example:
        >>> result = google_deep_search_tool.invoke({"query": "Python typing"})
        >>> # Returns: Search snippets + full Wikipedia/docs content
    """
    if not query or not query.strip():
        return "No results found for your query."

    # Step 1: Perform search and get sanitized results + URLs
    results = search(query, engine="google")

    # Check for errors
    if "error" in results:
        return f"Search failed: {results['error']}"

    # Get formatted snippets and source URLs
    formatted_snippets, source_urls = sanitize_search_results(results)

    # Step 2: Load full content from top URL
    if not source_urls:
        return formatted_snippets  # Return snippets only if no URLs

    target_url = source_urls[0]  # Use highest priority URL
