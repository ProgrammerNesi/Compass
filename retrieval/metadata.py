from retrieval.vector import vectorstore


def metadata_search(
    query: str,
    metadata_filter: dict,
    k: int = 10
):

    return vectorstore.similarity_search(
        query,
        k=k,
        filter=metadata_filter
    )