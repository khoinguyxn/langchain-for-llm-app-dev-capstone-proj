"""Research Agent - Mono-agent with multiple search and research tools."""

import json
from os import getenv
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import SystemMessage
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import BaseTool, StructuredTool, create_retriever_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable

from chroma import create_chroma_client


def _serialize_tool_result(result: Any) -> str:
    """
    Serialize tool results to a plain string format.

    Handles various result formats including structured objects, lists, and dicts.
    """
    if isinstance(result, str):
        return result
    elif isinstance(result, (dict, list)):
        return json.dumps(result, indent=2)
    else:
        return str(result)


def _wrap_mcp_tool(tool: BaseTool) -> BaseTool:
    """
    Wrap MCP tool to ensure results are properly serialized as strings.
    """
    original_func = tool.func if hasattr(tool, 'func') else tool._run
    original_afunc = tool.afunc if hasattr(tool, 'afunc') else tool._arun

    def wrapped_func(*args, **kwargs):
        result = original_func(*args, **kwargs)
        return _serialize_tool_result(result)

    async def wrapped_afunc(*args, **kwargs):
        result = await original_afunc(*args, **kwargs)
        return _serialize_tool_result(result)

    # Create a new tool with wrapped functions
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        func=wrapped_func,
        coroutine=wrapped_afunc,
        args_schema=tool.args_schema if hasattr(tool, 'args_schema') else None,
    )


@traceable(run_type="tool")
async def get_mcp_tools() -> list[BaseTool]:
    """
    Initialize MCP client and retrieve academia tools.

    Returns:
        List of MCP tools for arXiv research (wrapped for proper serialization).
    """
    mcp_clients = MultiServerMCPClient(
        {
            "academia": {
                "transport": "http",
                "url": "http://localhost:5056/mcp",
            }
        }
    )

    mcp_tools = await mcp_clients.get_tools()

    # Wrap each tool to ensure results are serialized as strings
    wrapped_tools = [_wrap_mcp_tool(tool) for tool in mcp_tools]

    return wrapped_tools


@traceable(run_type="retriever")
def create_local_retriever_tool() -> BaseTool:
    """
    Create a retriever tool from Chroma vector store.

    Returns:
        Retriever tool for searching local research papers.
    """
    chroma = create_chroma_client()
    retriever = chroma.as_retriever(search_kwargs={"k": 5})

    retriever_tool = create_retriever_tool(
        retriever,
        "search_local_papers",
        "Search through locally stored research papers for relevant context.",
    )

    return retriever_tool


@traceable(run_type="llm")
def get_llm_model() -> ChatOpenAI:
    """
    Initialize the LLM model for the research agent.

    Returns:
        Configured ChatOpenAI model.
    """
    model = ChatOpenAI(
        api_key=getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="mistralai/devstral-2512:free",
        temperature=0,
        max_tokens=2048,
    )

    return model


@traceable(run_type="prompt")
def get_system_prompt() -> SystemMessage:
    """
    Define the system prompt for the research agent.

    Returns:
        SystemMessage with agent instructions.
    """
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

    return system_prompt


@traceable(run_type="chain")
async def create_research_chain() -> RunnableSerializable:
    """
    Create a research chain that uses Chroma vector store and ChatOpenAI model.

    Returns:
        Agent that can execute tools and provide research answers.
    """
    # Load MCP tools
    mcp_tools = await get_mcp_tools()

    # Create local retriever tool
    retriever_tool = create_local_retriever_tool()

    # Combine all tools
    tools = [retriever_tool, *mcp_tools]

    # Initialize LLM model
    model = get_llm_model()

    # Get system prompt
    system_prompt = get_system_prompt()

    # Create the agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
        middleware=[SummarizationMiddleware(model=model, trigger=("tokens", 4000))],
    )

    return agent
