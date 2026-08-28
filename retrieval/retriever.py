"""
Dynamic collection context for retrieval.

This module holds a singleton reference to the currently active
collection's vector store and BM25 index. All retrieval modules
(vector, bm25, hybrid, etc.) read from here instead of creating
their own hardcoded instances.

When no collection is active, falls back to the legacy baseline
collection for backward compatibility with benchmark scripts.
"""

import pickle
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL,
    LEGACY_VECTOR_DB,
    LEGACY_BM25_FILE,
    COLLECTIONS_DIR,
)

# Shared embedding instance (cheap to reuse)
_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


class ActiveCollection:
    """Holds the retrievers for the currently active document collection."""

    def __init__(self, collection_id: str, vectorstore: Chroma, bm25_data: dict):
        self.collection_id = collection_id
        self.vectorstore = vectorstore
        self.bm25_documents = bm25_data.get("documents", [])
        self.bm25_index = bm25_data.get("bm25", None)


# Module-level singleton
_active: Optional[ActiveCollection] = None


def get_active() -> Optional[ActiveCollection]:
    """Return the currently active collection, or None."""
    return _active


def get_vectorstore() -> Chroma:
    """Return the active vectorstore, falling back to legacy baseline."""
    if _active is not None:
        return _active.vectorstore
    return _get_legacy_vectorstore()


def get_bm25():
    """Return the active BM25 index object, or raise if unavailable."""
    if _active is not None and _active.bm25_index is not None:
        return _active.bm25_index
    return _get_legacy_bm25()


def get_bm25_documents():
    """Return the active BM25 document list, or raise if unavailable."""
    if _active is not None and _active.bm25_documents:
        return _active.bm25_documents
    return _get_legacy_bm25_documents()


def set_active(collection_id: str):
    """
    Activate a collection so all retrieval tools use it.

    Looks for the collection's vector_db/ and bm25.pkl under
    collections/<collection_id>/.
    """
    global _active

    coll_dir = COLLECTIONS_DIR / collection_id
    vector_db_path = coll_dir / "vector_db"
    bm25_path = coll_dir / "bm25.pkl"

    if not vector_db_path.exists():
        raise FileNotFoundError(
            f"Vector DB not found for collection '{collection_id}' at {vector_db_path}"
        )

    vectorstore = Chroma(
        persist_directory=str(vector_db_path),
        collection_name="default",
        embedding_function=get_embeddings(),
    )

    bm25_data = {"documents": [], "bm25": None}
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)

    _active = ActiveCollection(
        collection_id=collection_id,
        vectorstore=vectorstore,
        bm25_data=bm25_data,
    )

    return _active


def set_active_legacy():
    """
    Activate the legacy baseline collection (vector_db/ + bm25.pkl).

    Used by benchmark scripts to preserve backward compatibility.
    """
    global _active

    if not LEGACY_VECTOR_DB.exists():
        raise FileNotFoundError(
            f"Legacy vector DB not found at {LEGACY_VECTOR_DB}. "
            "Run ingestion first."
        )

    vectorstore = Chroma(
        persist_directory=str(LEGACY_VECTOR_DB),
        collection_name="baseline",
        embedding_function=get_embeddings(),
    )

    bm25_data = {"documents": [], "bm25": None}
    if LEGACY_BM25_FILE.exists():
        with open(LEGACY_BM25_FILE, "rb") as f:
            bm25_data = pickle.load(f)

    _active = ActiveCollection(
        collection_id="baseline",
        vectorstore=vectorstore,
        bm25_data=bm25_data,
    )

    return _active


def clear_active():
    """Deactivate the current collection (falls back to legacy)."""
    global _active
    _active = None


# ── Legacy fallback helpers ──────────────────────────────────

_legacy_vectorstore: Optional[Chroma] = None


def _get_legacy_vectorstore() -> Chroma:
    global _legacy_vectorstore
    if _legacy_vectorstore is None:
        _legacy_vectorstore = Chroma(
            persist_directory=str(LEGACY_VECTOR_DB),
            collection_name="baseline",
            embedding_function=get_embeddings(),
        )
    return _legacy_vectorstore


def _get_legacy_bm25():
    if not LEGACY_BM25_FILE.exists():
        raise FileNotFoundError(
            f"Legacy BM25 index not found: {LEGACY_BM25_FILE}"
        )
    with open(LEGACY_BM25_FILE, "rb") as f:
        data = pickle.load(f)
    return data["bm25"]


def _get_legacy_bm25_documents():
    if not LEGACY_BM25_FILE.exists():
        raise FileNotFoundError(
            f"Legacy BM25 index not found: {LEGACY_BM25_FILE}"
        )
    with open(LEGACY_BM25_FILE, "rb") as f:
        data = pickle.load(f)
    return data["documents"]
