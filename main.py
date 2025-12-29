import asyncio

from research_chain import create_research_chain


async def main():
    chain = await create_research_chain()

    while True:
        question = input("Enter your research question (or 'exit' to quit): ")

        if question.lower() == "exit":
            break

        response = await chain.ainvoke(question)

        print(f"Response: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
