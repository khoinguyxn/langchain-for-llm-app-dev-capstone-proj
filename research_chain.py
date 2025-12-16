"""Research Agent - Mono-agent with multiple search and research tools."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_ollama import ChatOllama

from chroma import create_chroma_client


def create_research_chain() -> RunnableSerializable:
    """
    Create a research chain that uses Chroma vector store and Ollama chat model.

    Returns:
        Configured research chain as a RunnableSerializable.
    """
    model = ChatOllama(
        model="deepseek-r1:latest",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a research assistant with access to PDF research pappers for in-depth summarization.
        
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
