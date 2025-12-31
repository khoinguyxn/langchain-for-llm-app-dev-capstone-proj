import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import traceable

from research_chain import init_tools, research_agent


@traceable
async def main():
    # Initialize tools globally in research_chain module
    await init_tools()

    while True:
        question = input("\nEnter your research question (or 'exit' to quit): ")

        if question.lower() == "exit":
            break

        messages = [HumanMessage(content=question)]

        response = await research_agent.ainvoke(messages)

        print(f"\n{'='*80}")
        print(f"Response: {response.content}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    load_dotenv()

    asyncio.run(main())
