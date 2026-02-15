from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

persistent_directory = "db/chroma_pdf"
collection_name = "pdf_book"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    collection_name=collection_name,
    embedding_function=embedding_model
)

# Search for relevant documents
query = "What is protein?"

retriever = db.as_retriever(search_kwargs={"k": 2})

# threshold mode
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 2,
        "score_threshold": 0.3
    }
)

relevant_docs = retriever.invoke(query)

print(f"\nUser Query: {query}")

# Display results
print("\n--- Context ---\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"chunk {i} \n(Page {doc.metadata.get('page','NA')}):\n{doc.page_content}\n")
