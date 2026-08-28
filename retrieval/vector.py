"""
Vector similarity search — now uses the active collection dynamically.

Falls back to the legacy baseline collection when no collection is
explicitly activated (preserves benchmark compatibility).
"""

from retrieval.retriever import get_vectorstore


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

    vectorstore = get_vectorstore()

    return vectorstore.similarity_search(
        query,
        k=k
    )
