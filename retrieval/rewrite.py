"""
Query rewriting and rewritten search.

Uses the active collection dynamically for the downstream vector search.
"""

from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)


def rewrite_query(query: str) -> str:

    prompt = f"""
Rewrite this question into a retrieval-optimized
search query.

Preserve all important:

- people
- companies
- products
- dates
- prices
- contract information
- technical terms
- constraints

Do NOT answer the question.

Original question:
{query}

Return ONLY the rewritten query.
"""

    response = llm.invoke(prompt)

    content = response.content
    if isinstance(content, list):
        text_parts = [
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ]
        content = " ".join(text_parts)

    return content.strip()


def rewritten_search(query: str, k: int = 10):
    from retrieval.vector import vector_search

    rewritten = rewrite_query(query)
    return vector_search(rewritten, k=k)
