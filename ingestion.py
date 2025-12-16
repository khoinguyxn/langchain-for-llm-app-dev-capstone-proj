from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chroma import create_chroma_client

RESEARCH_PAPERS_DIR = Path("./data/research_papers")


def store_research_papers():
    """
    Store research papers from a PDF file into the Chroma vector store.
    """
    # Load PDF
    all_documents: List[Document] = []

    for file_path in RESEARCH_PAPERS_DIR.glob("*.pdf"):
        print(f"Loading PDF from {file_path}.")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Safe limit for nomic-embed-text
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(all_documents)

    print(f"Loaded and split into {len(chunks)} chunks (max 800 chars each).")

    # Store documents in Chroma vector store (using fresh client)
    print("Creating embeddings and storing in Chroma.")

    chroma = create_chroma_client()
    doc_ids = chroma.add_documents(chunks)

    print(f"Successfully stored {len(doc_ids)} documents.")


if __name__ == "__main__":
    store_research_papers()
