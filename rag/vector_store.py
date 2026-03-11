from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rag.embeddings import get_embeddings_model
from config.settings import settings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

# Initialize Qdrant Client (in-memory for demo, or URL for production)
_qdrant_client = QdrantClient(
    url=settings.QDRANT_URL if settings.QDRANT_URL != ":memory:" else None,
    location=":memory:" if settings.QDRANT_URL == ":memory:" else None,
    api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
)

def get_vector_store(collection_name: str = "psychology_knowledge") -> QdrantVectorStore:
    """
    Returns a QdrantVectorStore instance.
    Ensures the collection exists before returning.
    """
    embeddings = get_embeddings_model()
    sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")

    # In a real app we'd check if collection exists first,
    # but QdrantClient creates it via the langchain wrapper or we can explicitly create it.
    if not _qdrant_client.collection_exists(collection_name):
        _qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    return QdrantVectorStore(
        client=_qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embedding,
    )
