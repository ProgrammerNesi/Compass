"""
Metadata-filtered search.

Uses the active collection dynamically via the shared vectorstore.
"""

from retrieval.retriever import get_vectorstore


def metadata_search(
    query: str,
    metadata_filter: dict,
    k: int = 10
):

    vectorstore = get_vectorstore()

    return vectorstore.similarity_search(
        query,
        k=k,
        filter=metadata_filter
    )
