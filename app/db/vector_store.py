# === File: app/db/vector_store.py ===
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Setup Chroma vector store
def get_vector_store(persist_dir="storage/chroma"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(embedding_function=embedding_model, persist_directory=persist_dir)

def add_to_vector_store(documents, vector_store):
    from langchain.schema import Document
    docs = [Document(page_content=doc['content'], metadata={
        "source": doc['source'],
        "doc_type": doc['doc_type']
    }) for doc in documents]
    vector_store.add_documents(docs)
    vector_store.persist()

def search_vector_store(query, vector_store, k=5):
    return vector_store.similarity_search(query, k=k)
