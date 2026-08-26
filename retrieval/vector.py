from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).parent.parent
VECTOR_DB = BASE_DIR / "vector_db"


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma(
    persist_directory=str(VECTOR_DB),
    collection_name="baseline",
    embedding_function=embeddings
)


def vector_search(
    query: str,
    k: int = 10
):
    """
    Semantic similarity search.

    Best for:
    - normal questions
    - conceptual questions
    - semantically similar information
    """

    return vectorstore.similarity_search(
        query,
        k=k
    )