import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from rag.chunking import get_text_splitter
from rag.vector_store import get_vector_store

def ingest_document(file_path: str, collection_name: str = "psychology_knowledge"):
    """
    Loads, chunks, and stores a document in the vector store.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Select loader based on extension
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        # Default to text
        loader = TextLoader(file_path)
        
    docs = loader.load()
    
    print(f"Loaded {len(docs)} documents from {file_path}")
    
    text_splitter = get_text_splitter()
    chunks = text_splitter.split_documents(docs)
    
    print(f"Created {len(chunks)} chunks. Ingesting to Qdrant...")
    
    vector_store = get_vector_store(collection_name)
    vector_store.add_documents(chunks)
    
    print("Ingestion complete.")

if __name__ == "__main__":
    # Example usage
    # ingest_document("path/to/research_paper.pdf")
    pass
