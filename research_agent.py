"""Research Agent - Mono-agent with multiple search and research tools."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from chroma import create_chroma_client

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

research_agent = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

result = research_agent.invoke("Summarize the key findings of the research paper.")

print(result)
