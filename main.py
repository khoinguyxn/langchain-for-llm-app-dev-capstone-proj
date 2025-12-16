from research_chain import create_research_chain


def main():
    chain = create_research_chain()

    while True:
        question = input("Enter your research question (or 'exit' to quit): ")

        if question.lower() == "exit":
            break

        response = chain.invoke(question)

        print(f"Response: {response}\n")


if __name__ == "__main__":
    main()
