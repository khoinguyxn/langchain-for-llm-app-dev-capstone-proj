import asyncio

from langchain_core.runnables import RunnableConfig

from research_chain import create_research_chain

RUNNABLE_CONFIG: RunnableConfig = {"configurable": {"thread_id": 1}}


async def main():
    chain = await create_research_chain()

    while True:
        question = input("\nEnter your research question (or 'exit' to quit): ")

        if question.lower() == "exit":
            break

        # create_agent returns an agent that expects messages format
        response = await chain.ainvoke(
            {"messages": [{"role": "user", "content": question}]}, RUNNABLE_CONFIG
        )

        print(f"\n{'='*80}")
        print(f"Answer: {response['messages'][-1].content}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
