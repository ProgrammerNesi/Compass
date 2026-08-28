"""
Multi-query expansion and MMR search.

Uses the active collection dynamically via the shared vectorstore.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path

from retrieval.retriever import get_vectorstore
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)


# ============================================================
# Multi Query
# ============================================================

def generate_queries(query: str, number: int = 3):

    prompt = f"""
Generate {number} different retrieval queries
for the following question.

Each query should focus on a different aspect
of the question.

Question:
{query}

Return exactly {number} queries.
One query per line.
"""

    response = llm.invoke(prompt)

    content = response.content
    if isinstance(content, list):
        text_parts = [
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ]
        content = " ".join(text_parts)

    queries = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    return queries[:number]


def multi_query_search(query: str, k: int = 10):

    vectorstore = get_vectorstore()
    queries = generate_queries(query)
    results = []
    seen = set()

    for search_query in queries:
        docs = vectorstore.similarity_search(search_query, k=k)
        for doc in docs:
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                results.append(doc)

    return results[:k]


# ============================================================
# MMR
# ============================================================

def mmr_search(query: str, k: int = 10):

    vectorstore = get_vectorstore()

    return vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=30,
        lambda_mult=0.5
    )
