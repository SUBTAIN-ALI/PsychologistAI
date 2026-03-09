from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter for document chunking.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.VECTOR_CHUNK_SIZE,
        chunk_overlap=settings.VECTOR_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
