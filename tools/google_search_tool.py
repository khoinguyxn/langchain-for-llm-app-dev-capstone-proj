"""Google Search tool with LLM-optimized output sanitization.

Wraps SerpAPI to provide clean, token-efficient search results for LangChain agents.
Prioritizes high-quality information sources and removes metadata/tracking data.

Environment:
    SERPAPI_API_KEY: SerpAPI authentication key (required)

Dependencies:
    serpapi>=0.1.5, python-dotenv>=1.2.1
"""

from langchain.tools import tool
from .config import get_serpapi_client


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
        and top 3 organic results (title, snippet, source). Returns fallback
        message if no results found.

    Raises:
        Exception: Generic handler for API/network failures.

    Example:
        >>> result = google_search("Python async programming")
        >>> print(result)
        Knowledge Graph:
        **Asynchronous I/O** (Programming paradigm)
        ...
    """
    if not query or not query.strip():
        return "No results found for your query."

    query = query.strip()[:500]

    try:
        params = {
            "engine": "google",
            "q": query,
        }

        client = get_serpapi_client()
        results = client.search(params)
    except Exception as e:
        print(f"SerpAPI search error: {e}")

        return f"Search failed: Unexpected error occurred."

    sanitized_results = []

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

            sanitized_results.append(
                f"Result {i}:\nTitle: {title}\nSnippet: {snippet}\nSource: {link}"
            )

    # 4. Combine all results
    if sanitized_results:
        return "\n\n".join(sanitized_results)

    return "No results found for your query."
