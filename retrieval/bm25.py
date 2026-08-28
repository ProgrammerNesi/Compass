"""
BM25 lexical search — now uses the active collection dynamically.

Falls back to the legacy baseline BM25 index when no collection is
explicitly activated (preserves benchmark compatibility).
"""

from retrieval.retriever import get_bm25, get_bm25_documents


class BM25Retriever:
    """Wraps the active collection's BM25 index."""

    def __init__(self):
        self.bm25 = get_bm25()
        self.documents = get_bm25_documents()

    def search(self, query: str, k: int = 10):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        return [self.documents[i] for i in ranked_indices[:k]]


bm25_retriever = None


def bm25_search(query: str, k: int = 10):
    global bm25_retriever

    # Re-initialize if the active collection may have changed
    bm25_retriever = BM25Retriever()
    return bm25_retriever.search(query, k)
