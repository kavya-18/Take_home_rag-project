import os
import textwrap

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_pdf(pdf_path="data/book.pdf"):
    """Load PDF pages from the given path"""
    print(f"\nLoading PDF from {pdf_path}...")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The directory {pdf_path} does not exist. Please create it and add your company files.")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No pages loaded from PDF: {pdf_path}")

    print(f"\nPages loaded: {len(documents)}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split PDF documents into smaller chunks with overlap"""
    print("\n Splitting PDF into chunks...\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"\nPage: {chunk.metadata.get('page', 'NA')}")
            print(f"\nLength: {len(chunk.page_content)} characters")
            print("\nContent preview:")
            print(textwrap.fill(chunk.page_content[:500], width=110))
            print("-" * 50)

        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks\n")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_pdf", collection_name="pdf_book"):
    """Create and persist Chroma vector store"""
    print("\nCreating embeddings and storing in ChromaDB...\n")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("--- Creating vector store ---\n")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("--- Finished creating vector store ---\n")

    print(f"\nVector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("\n=== PDF RAG Ingestion Pipeline (Local) ===\n")

    pdf_path = "data/book.pdf"
    persist_directory = "db/chroma_pdf"
    collection_name = "pdf_book"

    # Check if vector store already exists
    if os.path.exists(persist_directory):
        print("\n Vector store already exists. No need to re-process PDF.\n")

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            collection_name="pdf_book",
            embedding_function=embedding_model
        )

        print(f"\nLoaded existing vector store with {vectorstore._collection.count()} chunks\n")
        return vectorstore

    print("Persistent directory does not exist. Initializing vector store...\n")

    # Step 1: Load PDF
    documents = load_pdf(pdf_path=pdf_path)

    # Step 2: Split into chunks
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=0)

    # Step 3: Create vector store
    vectorstore = create_vector_store(chunks, persist_directory=persist_directory, collection_name="pdf_book")

    print("\n Ingestion complete! Your PDF is now ready for RAG queries.\n")
    return vectorstore


if __name__ == "__main__":
    main()
