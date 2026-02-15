from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

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
query = "how much water should one drink?"

retriever = db.as_retriever(search_kwargs={"k": 2})
relevant_docs = retriever.invoke(query)

print(f"\nUser Query: {query}\n")
print("\n--- Context ---\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i} (Page {doc.metadata.get('page','NA')}):\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = f"""
You are a helpful assistant.
Answer the question using only the information from the documents below.
If the answer is not in the documents, say:
"NOT AVAILABLE IN THE PROVIDED PDF CONTEXT."

QUESTION:
{query}

DOCUMENTS:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

ANSWER:
""".strip()

# Create a model
model = ChatOllama(model="phi3:mini", temperature=0)

# Invoke the model with the combined input
result = model.invoke(combined_input)

print("\n--- Generated Response ---\n")
print(result.content)
