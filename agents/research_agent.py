from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool

deepseek = ChatOllama(
    model="qwen3:0.6b",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


agent = create_agent(model=deepseek, tools=[get_weather])

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather in Ho Chi Minh City?"}
        ]
    }
)

print(response)
