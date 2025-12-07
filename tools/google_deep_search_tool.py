from langchain.tools import tool
from libs.serpapi.search import search
from typing import Optional


@tool
def google_deep_search_tool(query: str) -> Optional[str]:
    """Perform deep research by searching and loading full page content.

    Searches Google, extracts URLs from knowledge graph or top results,
    loads the full web page content, and returns comprehensive information.

    Use this when you need detailed, in-depth content beyond snippets.

    Args:
        query: Search query string.

    Returns:
        Full page content from the top relevant source.
    """
    if not query or not query.strip():
        return "No results found for your query."

    try:
        results = search(query, engine="google")
    except Exception as e:
        print(f"SerpAPI search error: {e}")
        return f"Search failed: Unexpected error occurred."

    # Try to extract URL from knowledge graph first
    target_url = None

    if "knowledge_graph" in results and isinstance(results["knowledge_graph"], dict):
        kg = results["knowledge_graph"]

        if "source" in kg and isinstance(kg["source"], dict):
            source = kg["source"]
            target_url = source.get("link")

    # Fallback to first organic result if no knowledge graph URL
    if not target_url and "organic_results" in results:
        organic = results["organic_results"]

        if organic and isinstance(results["organic_results"], list):
            target_url = organic[0].get("link")

    if not target_url:
        return None
