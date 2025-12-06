from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools.google_search_tool import google_search_tool
from tools.google_scholar_tool import google_scholar_tool

qwen3 = ChatOllama(
    model="qwen3:14b",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


agent = create_agent(model=qwen3, tools=[google_search_tool, google_scholar_tool])

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the current prime minister of Vietnam?",
            }
        ]
    }
)

print(response)
