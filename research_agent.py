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
from models.research_answer import ResearchAnswer

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


@task
@traceable(run_type="llm")
async def format_final_answer(
    conversation_history: List[BaseMessage],
) -> ResearchAnswer:
    """
    Format the final answer into structured ResearchAnswer format.

    Args:
        conversation_history: Full conversation including tool results

    Returns:
        Structured research answer with citations
    """
    llm = ChatOpenAI(
        api_key=getenv("OPENROUTER_API_KEY"),
        base_url=getenv("OPENROUTER_BASE_URL"),
        model=getenv("OPENROUTER_MODEL"),
        temperature=0,
        max_tokens=2048,
        model_kwargs={"response_format": {"type": "json_object"}},
    ).with_structured_output(ResearchAnswer, include_raw=False)

    formatting_prompt = SystemMessage(
        content="""Based on the conversation and research gathered, format a final research answer.

You MUST respond with valid JSON ONLY. Do not include markdown code blocks or any other text.

Required JSON structure (remove all comments before responding):
{
  "answer": "your comprehensive research answer here",
  "confidence_score": 0.85,
  "citations": [
    {
      "title": "Paper Title",
      "authors": ["Author One", "Author Two"],
      "year": 2024,
      "doi": "10.1234/example",
      "url": "https://arxiv.org/abs/1234.5678",
      "page_numbers": null
    }
  ]
}

Rules:
- "answer" must be a string with the research summary
- "confidence_score" must be a float between 0.0 and 1.0
- "citations" must be an array of citation objects
- "authors" must be an array of strings, NOT a single string
- Use null for missing fields (year, page_numbers, doi, url)
- Extract citations from arXiv search results in the conversation
- For arXiv papers, URL format: https://arxiv.org/abs/{arxiv_id}
"""
    )

    try:
        result = await llm.ainvoke([formatting_prompt] + conversation_history)
        return result
    except Exception as e:
        # Fallback with minimal structure
        print(f"Warning: Structured output failed: {e}")

        return ResearchAnswer(
            answer="Unable to format structured answer due to parsing error. Please check the conversation history.",
            confidence_score=0.0,
            citations=[],
        )


@entrypoint(checkpointer=False)
@traceable(run_type="chain")
async def research_agent(messages: List[BaseMessage]) -> ResearchAnswer:
    """
    Research agent entrypoint using LangGraph Functional API.

    Args:
        messages: List of messages (conversation history)

    Returns:
        Structured research answer with citations
    """
    MAX_ITERATIONS = 10

    for _ in range(MAX_ITERATIONS):
        # Call model
        ai_message = await call_model(messages)

        # If no tool calls, format and return final answer
        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
            # Add final AI message to history
            messages.append(ai_message)
            # Format into structured output
            return await format_final_answer(messages)

        # Execute tools
        tool_messages = await call_tools(ai_message)

        # Add to message history
        messages.append(ai_message)
        messages.extend(tool_messages)

    # Max iterations - still format what we have
    messages.append(
        AIMessage(
            content="Max iterations reached. Providing best answer with available information."
        )
    )

    return await format_final_answer(messages)
