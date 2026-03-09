from langchain_core.vectorstores import VectorStoreRetriever
from rag.vector_store import get_vector_store

def get_retriever(collection_name: str = "psychology_knowledge", k: int = 4) -> VectorStoreRetriever:
    """
    Returns a configured retriever from the vector store.
    """
    vector_store = get_vector_store(collection_name)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
