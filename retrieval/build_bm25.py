import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from ingestion.ingestion import fetch_documents, create_chunks


BASE_DIR = Path(__file__).parent.parent

BM25_FILE = (
    BASE_DIR
    / "vector_db"
    / "bm25.pkl"
)


def build_bm25_index():

    documents = fetch_documents()

    chunks = create_chunks(
        documents
    )

    tokenized_documents = [
        doc.page_content.lower().split()
        for doc in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    with open(
        BM25_FILE,
        "wb"
    ) as f:

        pickle.dump(
            {
                "documents": chunks,
                "bm25": bm25
            },
            f
        )

    print(
        f"BM25 index created with "
        f"{len(chunks)} chunks."
    )


if __name__ == "__main__":
    build_bm25_index()