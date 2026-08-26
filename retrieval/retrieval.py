from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).parent.parent
DB_NAME = BASE_DIR / "vector_db"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

RETRIEVAL_K = 10


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


vectorstore = Chroma(
    persist_directory=str(DB_NAME),
    collection_name="baseline",
    embedding_function=embeddings
)


retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": RETRIEVAL_K
    }
)


def retrieve(question: str):

    docs = retriever.invoke(question)

    return docs


if __name__ == "__main__":

    question = "Who founded Insurellm?"

    docs = retrieve(question)

    print(f"\nRetrieved {len(docs)} documents\n")

    for i, doc in enumerate(docs, 1):

        print("=" * 80)
        print(f"RESULT {i}")
        print("=" * 80)

        print(
            "Source:",
            doc.metadata.get("source")
        )

        print(doc.page_content[:1000])