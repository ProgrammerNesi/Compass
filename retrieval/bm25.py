import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).parent.parent

BM25_FILE = (
    BASE_DIR
    / "vector_db"
    / "bm25.pkl"
)


class BM25Retriever:

    def __init__(self):

        if not BM25_FILE.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {BM25_FILE}\n"
                "Run build_bm25_index() first."
            )

        with open(
            BM25_FILE,
            "rb"
        ) as f:

            data = pickle.load(f)

        self.documents = data["documents"]
        self.bm25 = data["bm25"]


    def search(
        self,
        query: str,
        k: int = 10
    ):

        tokens = query.lower().split()

        scores = self.bm25.get_scores(
            tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [
            self.documents[i]
            for i in ranked_indices[:k]
        ]


bm25_retriever = None


def bm25_search(
    query: str,
    k: int = 10
):

    global bm25_retriever

    if bm25_retriever is None:

        bm25_retriever = BM25Retriever()

    return bm25_retriever.search(
        query,
        k
    )