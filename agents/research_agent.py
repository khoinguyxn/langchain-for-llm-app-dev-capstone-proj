from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools.google_search_tool import google_search_tool

deepseek = ChatOllama(
    model="qwen3:0.6b",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


agent = create_agent(model=deepseek, tools=[google_search_tool])

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
