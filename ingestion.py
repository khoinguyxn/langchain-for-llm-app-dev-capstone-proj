from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chroma import create_chroma_client


def store_research_papers(file_path: str):
    """
    Store research papers from a PDF file into the Chroma vector store.

    Args:
        file_path (str): The path to the PDF file containing research papers.
    """
    # Load PDF
    print(f"Loading PDF from {file_path}.")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Safe limit for nomic-embed-text
        chunk_overlap=100,
    )
    documents = text_splitter.split_documents(documents)

    print(f"Loaded and split into {len(documents)} chunks (max 800 chars each).")

    # Store documents in Chroma vector store (using fresh client)
    print("Creating embeddings and storing in Chroma.")

    chroma = create_chroma_client()
    doc_ids = chroma.add_documents(documents)

    print(f"Successfully stored {len(doc_ids)} documents.")


if __name__ == "__main__":
    FILE_PATH = "./data/paper_1.pdf"
    store_research_papers(FILE_PATH)
