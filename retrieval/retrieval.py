"""
Standalone baseline retriever — uses the active collection dynamically.

Falls back to legacy baseline when no collection is active.
"""

from retrieval.retriever import get_vectorstore
from config import RETRIEVAL_K


def retrieve(question: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )
    return retriever.invoke(question)


if __name__ == "__main__":

    question = "Who founded Insurellm?"

    docs = retrieve(question)

    print(f"\nRetrieved {len(docs)} documents\n")

    for i, doc in enumerate(docs, 1):
        print("=" * 80)
        print(f"RESULT {i}")
        print("=" * 80)
        print("Source:", doc.metadata.get("source"))
        print(doc.page_content[:1000])
