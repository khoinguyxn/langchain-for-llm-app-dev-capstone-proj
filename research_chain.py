"""Research Agent - Mono-agent with multiple search and research tools."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from chroma import create_chroma_client
from models.research_answer import ResearchAnswer


async def create_research_chain() -> RunnableSerializable:
    """
    Create a research chain that uses Chroma vector store and Ollama chat model.

    Returns:
        Configured research chain as a RunnableSerializable.
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

    model = ChatOllama(
        model="deepseek-r1:latest",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    ).bind_tools(mcp_tools)

    prompt = ChatPromptTemplate.from_template(
        """You are a research assistant with access to PDF research papers for in-depth summarization.
        
        You have access to academia research tools that can:
        - arxiv_search: Query arXiv with field-specific queries and filters.
        - arxiv_download: Fetch a paper by ID and convert to structured text (HTML/PDF modes).
        
        Always provide clear, concise summary.
        
        Context: {context}
        Question: {question}
    """
    )

    chroma = create_chroma_client()
    retriever = chroma.as_retriever(search_kwargs={"k": 5})

    return (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )
