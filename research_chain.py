"""Research Agent - Mono-agent with multiple search and research tools."""

from os import getenv

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import SystemMessage
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import create_retriever_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from chroma import create_chroma_client

load_dotenv()


async def create_research_chain() -> RunnableSerializable:
    """
    Create a research chain that uses Chroma vector store and ChatOpenAI model.

    Returns:
        AgentExecutor (is a RunnableSerializable) that executes tools.
    """
    # Load MCP tools
    mcp_clients = MultiServerMCPClient(
        {
            "academia": {
                "transport": "http",
                "url": "http://localhost:5056/mcp",
            }
        }
    )

    mcp_tools = await mcp_clients.get_tools()

    # Create retriever tool from Chroma
    chroma = create_chroma_client()
    retriever = chroma.as_retriever(search_kwargs={"k": 5})

    retriever_tool = create_retriever_tool(
        retriever,
        "search_local_papers",
        "Search through locally stored research papers for relevant context.",
    )

    tools = [retriever_tool, *mcp_tools]

    # Initialize LLM model
    model = ChatOpenAI(
        api_key=getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="mistralai/devstral-2512:free",
        temperature=0,
        max_tokens=2048,
    )

    # Define system prompt
    system_prompt = SystemMessage(
        content="""
        You are a research assistant with access to both local PDF research papers and arXiv.

        Available tools:
        - search_local_papers: Search locally stored research papers
        - arxiv_search: Query arXiv with field-specific queries
        - arxiv_download: Fetch papers by ID and convert to text
        - And 10+ other academia tools

        Strategy:
        1. Search local papers first for relevant context
        2. Use arXiv search if you need additional papers
        3. Provide clear, concise summaries
        """
    )

    # Create the agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
        middleware=[SummarizationMiddleware(model=model, trigger=("tokens", 4000))],
    )

    return agent
