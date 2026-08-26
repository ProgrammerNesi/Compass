from langchain_core.tools import tool

from retrieval.vector import vector_search
from retrieval.bm25 import bm25_search
from retrieval.hybrid import hybrid_search

from retrieval.rewrite import rewritten_search

from retrieval.expansion import (
    multi_query_search,
    mmr_search
)

from retrieval.reranker import (
    reranker_search
)

from retrieval.metadata import (
    metadata_search
)



# ============================================================
# Helper
# ============================================================

def serialize_documents(
    documents
):

    return [
        {
            "source": doc.metadata.get(
                "source"
            ),
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]


# ============================================================
# 1. VECTOR SEARCH
# ============================================================

@tool
def vector_search_tool(
    query: str
):
    """
    Semantic vector search.

    Use for normal conceptual or semantic questions.
    """

    docs = vector_search(
        query,
        k=10
    )

    return serialize_documents(
        docs
    )


# ============================================================
# 2. BM25 SEARCH
# ============================================================

@tool
def bm25_search_tool(
    query: str
):
    """
    Lexical search using BM25.

    Use when the question contains exact names,
    product names, prices, dates, contracts or terminology.
    """

    docs = bm25_search(
        query,
        k=10
    )

    return serialize_documents(
        docs
    )


# ============================================================
# 3. HYBRID SEARCH
# ============================================================

@tool
def hybrid_search_tool(
    query: str
):
    """
    Combines vector semantic search and BM25 lexical search.

    Use when both semantic meaning and exact terminology
    are important.
    """

    docs = hybrid_search(
        query,
        k=10
    )

    return serialize_documents(
        docs
    )


# ============================================================
# 4. QUERY REWRITE
# ============================================================

@tool
def query_rewrite_tool(
    query: str
):
    """
    Rewrite a poorly phrased user query into a
    retrieval-optimized query.
    """

    return rewritten_search(
        query,
        k=10
    )


# ============================================================
# 5. MULTI-QUERY
# ============================================================

@tool
def multi_query_tool(
    query: str
):
    """
    Generate multiple search queries and combine their
    retrieved results.

    Best for complex questions containing multiple aspects.
    """

    docs = multi_query_search(
        query,
        k=10
    )

    return serialize_documents(
        docs
    )


# ============================================================
# 6. RERANKER
# ============================================================

@tool
def reranker_tool(
    query: str
):
    """
    Retrieve a larger candidate pool and rerank it using
    a cross-encoder.

    Useful when relevant documents are retrieved but
    appear at poor ranks.
    """

    docs = reranker_search(
        query,
        k=10
    )

    return serialize_documents(
        docs
    )


# ============================================================
# 7. MMR
# ============================================================

@tool
def mmr_tool(
    query: str
):
    """
    Maximal Marginal Relevance retrieval.

    Retrieves relevant but diverse documents.

    Useful for multi-document and comparison questions.
    """

    docs = mmr_search(
        query,
        k=10
    )

    return serialize_documents(
        docs
    )


# ============================================================
# 8. METADATA FILTER
# ============================================================

@tool
def metadata_filter_tool(
    query: str,
    document_type: str
):
    """
    Restrict retrieval to a particular document type.

    Examples:
    company
    products
    contracts
    employees
    """

    docs = metadata_search(
        query,
        {
            "doc_type": document_type
        },
        k=10
    )

    return serialize_documents(
        docs
    )


# ============================================================
# 9. PARENT DOCUMENT
# ============================================================


# ============================================================
# ALL RETRIEVAL TOOLS
# ============================================================

RETRIEVAL_TOOLS = [

    vector_search_tool,

    bm25_search_tool,

    hybrid_search_tool,

    query_rewrite_tool,

    multi_query_tool,

    reranker_tool,

    mmr_tool,

    metadata_filter_tool,

]