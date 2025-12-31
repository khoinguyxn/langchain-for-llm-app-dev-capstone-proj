"""Research Agent - LangGraph Functional API implementation."""

from os import getenv
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, create_retriever_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task
from langsmith import traceable

from chroma import create_chroma_client

SYSTEM_PROMPT = """You are a research assistant with access to both local PDF research papers and arXiv.

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

TOOLS: List[BaseTool] = []


@traceable(run_type="tool")
async def init_tools() -> None:
    """Initialize global TOOLS: MCP academia tools + local retriever."""
    # Load MCP tools
    mcp_client = MultiServerMCPClient(
        {
            "academia": {
                "transport": "http",
                "url": "http://localhost:5056/mcp",
            }
        }
    )
    mcp_tools = await mcp_client.get_tools()

    # Create local retriever
    chroma = create_chroma_client()
    retriever = chroma.as_retriever(search_kwargs={"k": 5})
    local_tool = create_retriever_tool(
        retriever,
        "search_local_papers",
        "Search through locally stored research papers for relevant context.",
    )

    TOOLS.extend([local_tool, *mcp_tools])


@task
@traceable(run_type="llm")
async def call_model(messages: List[BaseMessage]) -> AIMessage:
    """Call LLM with tools bound."""
    llm = ChatOpenAI(
        api_key=getenv("OPENROUTER_API_KEY"),
        base_url=getenv("OPENROUTER_BASE_URL"),
        model=getenv("OPENROUTER_MODEL"),
        temperature=0,
        max_tokens=2048,
    ).bind_tools(TOOLS)

    return await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)


@task
@traceable(run_type="tool")
async def call_tools(ai_message: AIMessage) -> List[ToolMessage]:
    """Execute tool calls from AI message."""
    tools_by_name = {tool.name: tool for tool in TOOLS}
    tool_messages = []

    for tool_call in ai_message.tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        if not tool:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool {tool_call['name']} not found",
                    tool_call_id=tool_call["id"],
                )
            )
            continue

        try:
            result = await tool.ainvoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        except Exception as e:
            tool_messages.append(
                ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call["id"])
            )

    return tool_messages


@entrypoint(checkpointer=False)
@traceable(run_type="chain")
async def research_agent(messages: List[BaseMessage]) -> AIMessage:
    """
    Research agent entrypoint using LangGraph Functional API.

    Args:
        messages: List of messages (conversation history)

    Returns:
        Final AI message response
    """
    MAX_ITERATIONS = 10

    for _ in range(MAX_ITERATIONS):
        # Call model
        ai_message = await call_model(messages)

        # If no tool calls, return final answer
        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
            return ai_message

        # Execute tools
        tool_messages = await call_tools(ai_message)

        # Add to message history
        messages.append(ai_message)
        messages.extend(tool_messages)

    return AIMessage(content="Max iterations reached.")
