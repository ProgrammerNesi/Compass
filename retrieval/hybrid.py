"""
Hybrid search: vector + BM25 with Reciprocal Rank Fusion.

Uses the active collection dynamically.
"""

from retrieval.vector import vector_search
from retrieval.bm25 import bm25_search


def hybrid_search(
    query: str,
    k: int = 10
):
    """
    Combines semantic and lexical retrieval.

    Vector search finds meaning.
    BM25 finds exact terms.
    """

    vector_docs = vector_search(query, k=k)
    bm25_docs = bm25_search(query, k=k)

    combined = []
    seen = set()
    scores = {}

    all_docs = vector_docs + bm25_docs

    for rank, doc in enumerate(vector_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (60 + rank)

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (60 + rank)

    doc_map = {doc.page_content: doc for doc in all_docs}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for content, _ in ranked:
        if content not in seen:
            seen.add(content)
            combined.append(doc_map[content])

    return combined[:k]
