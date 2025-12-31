import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import traceable

from models.research_answer import ResearchAnswer
from research_agent import init_tools, research_agent


@traceable
async def main():
    # Initialize tools globally in research_chain module
    await init_tools()

    while True:
        question = input("\nEnter your research question (or 'exit' to quit): ")

        if question.lower() == "exit":
            break

        messages = [HumanMessage(content=question)]

        response: ResearchAnswer = await research_agent.ainvoke(messages)

        print(f"\n{'='*80}")
        print(f"Response:\n{response.answer}\n")
        print(f"Confidence: {response.confidence_score:.2%}\n")
        print(f"Citations: ({len(response.citations)}):")

        for i, citation in enumerate(response.citations, 1):
            print(f"  [{i}] {citation.title} ({citation.year})")
            print(f"      Authors: {', '.join(citation.authors)}")
            print(f"      DOI: {citation.doi}")
            print(f"      URL: {citation.url}\n")

        print(f"{'='*80}\n")


if __name__ == "__main__":
    load_dotenv()

    asyncio.run(main())
