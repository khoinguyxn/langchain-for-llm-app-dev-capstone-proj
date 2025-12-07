"""Shared configuration and client initialization for search tools.

Centralizes environment variable loading and SerpAPI client setup to avoid
code duplication across multiple tool modules.
"""

import os
from dotenv import load_dotenv
import serpapi
from typing import Optional

# Load environment variables once at module import
load_dotenv()


def get_serpapi_key() -> str:
    """Retrieve and validate SERPAPI_API_KEY from environment.

    Returns:
        API key string.

    Raises:
        ValueError: If SERPAPI_API_KEY environment variable is not set.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError(
            "SERPAPI_API_KEY environment variable is required but not set. "
            "Please add it to your .env file."
        )
        
    return api_key


# Singleton SerpAPI client instance
_serpapi_client: Optional[serpapi.Client] = None


def get_serpapi_client() -> serpapi.Client:
    """Get or create singleton SerpAPI client instance.

    Lazy-loads the client on first access to ensure API key is validated
    only when needed.

    Returns:
        Configured SerpAPI client instance.

    Raises:
        ValueError: If SERPAPI_API_KEY is not set.
    """
    global _serpapi_client
    
    if _serpapi_client is None:
        api_key = get_serpapi_key()
        _serpapi_client = serpapi.Client(api_key=api_key)
    
    return _serpapi_client
