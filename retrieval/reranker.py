"""
Cross-encoder reranking.

Uses the active collection dynamically for the upstream vector search.
"""

from sentence_transformers import CrossEncoder

from retrieval.vector import vector_search
from config import RERANKER_MODEL


reranker = CrossEncoder(RERANKER_MODEL)


def rerank_documents(query: str, documents, k: int = 10):

    if not documents:
        return []

    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for score, doc in ranked[:k]]


def reranker_search(query: str, k: int = 10):

    candidates = vector_search(query, k=30)
    return rerank_documents(query, candidates, k)
