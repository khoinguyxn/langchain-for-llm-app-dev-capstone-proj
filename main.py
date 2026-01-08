"""Main entry point for Research Agent - supports both text and voice input modes."""

import asyncio
import subprocess

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import traceable

from models.research_answer import ResearchAnswer
from research_agent import init_tools, research_agent


@traceable
async def text_mode():
    """Interactive text-based research mode."""
    print("\n" + "=" * 80)
    print("Text Research Mode - Enter research questions to get answers with citations")
    print("=" * 80 + "\n")

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


@traceable
async def voice_mode():
    """LiveKit voice-based research mode."""
    print("\n" + "=" * 80)
    print("Voice Research Mode - LiveKit Agent Server")
    print("Connect to your LiveKit room to start asking research questions via voice.")
    print("=" * 80 + "\n")

    from libs.langsmith import setup_tracing

    setup_tracing()

    subprocess.run(["uv", "run", "-m", "voice_research_agent", "console"])


async def main():
    """Main entry point with mode selection."""
    print("\n" + "=" * 80)
    print("Research Agent - AI-Powered Research Assistant")
    print("=" * 80)
    print("\nAvailable modes:")
    print("  1. Text Input  - Interactive CLI for text-based research queries")
    print("  2. Voice Input - LiveKit-powered voice research assistant")
    print()

    await init_tools()

    mode = int(input("Select mode (1 or 2): "))

    match mode:
        case 1:
            await text_mode()
        case 2:
            await voice_mode()
        case _:
            print("Invalid mode selected. Please choose 1 or 2.")


if __name__ == "__main__":
    load_dotenv()

    asyncio.run(main())
