from langchain_openai import OpenAIEmbeddings
from config.settings import settings

def get_embeddings_model() -> OpenAIEmbeddings:
    """
    Returns the configured OpenAIEmbeddings model.
    """
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY
    )
