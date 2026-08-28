"""
Collection manager for document isolation.

Each uploaded document set becomes a named collection stored under
collections/<collection_id>/ with its own vector_db/ and bm25.pkl.

Collections are completely independent — queries against one never
touch another collection's data.
"""

import json
import pickle
import uuid
from pathlib import Path
from typing import List, Optional, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import (
    COLLECTIONS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
)
from ingestion.loaders import load_files
from retrieval.retriever import get_embeddings


REGISTRY_FILE = COLLECTIONS_DIR / "registry.json"


def _load_registry() -> Dict:
    """Load the collection registry from disk."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {"collections": {}}


def _save_registry(registry: Dict):
    """Save the collection registry to disk."""
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def list_collections() -> List[Dict]:
    """Return metadata for all registered collections."""
    registry = _load_registry()
    return [
        {"id": cid, **info}
        for cid, info in registry.get("collections", {}).items()
    ]


def get_collection_info(collection_id: str) -> Optional[Dict]:
    """Return metadata for a specific collection."""
    registry = _load_registry()
    return registry.get("collections", {}).get(collection_id)


def create_collection(
    collection_id: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Create a new empty collection. Returns the collection ID.
    If collection_id is not provided, a UUID is generated.
    """
    if collection_id is None:
        collection_id = str(uuid.uuid4())[:8]

    coll_dir = COLLECTIONS_DIR / collection_id
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "vector_db").mkdir(exist_ok=True)

    registry = _load_registry()
    registry["collections"][collection_id] = {
        "name": name or collection_id,
        "created_at": str(Path(__file__)),  # placeholder
        "document_count": 0,
        "chunk_count": 0,
        "status": "empty",
        "files": [],
    }
    _save_registry(registry)

    return collection_id


def delete_collection(collection_id: str):
    """Delete a collection and its data."""
    import shutil

    coll_dir = COLLECTIONS_DIR / collection_id
    if coll_dir.exists():
        shutil.rmtree(coll_dir)

    registry = _load_registry()
    registry["collections"].pop(collection_id, None)
    _save_registry(registry)


def ingest_files(
    collection_id: str,
    file_paths: List[Path],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Dict:
    """
    Ingest files into an existing collection.

    1. Load files via multi-format loaders
    2. Chunk the documents
    3. Embed and store in Chroma
    4. Build BM25 index
    5. Update registry

    Returns a summary dict.
    """
    coll_dir = COLLECTIONS_DIR / collection_id
    if not coll_dir.exists():
        create_collection(collection_id)

    vector_db_path = coll_dir / "vector_db"
    bm25_path = coll_dir / "bm25.pkl"

    # 1. Load
    documents = load_files(file_paths)
    if not documents:
        return {"error": "No documents could be loaded from the provided files."}

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed + store in Chroma
    # Delete existing collection if present
    try:
        existing = Chroma(
            persist_directory=str(vector_db_path),
            collection_name="default",
            embedding_function=get_embeddings(),
        )
        existing.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(vector_db_path),
        collection_name="default",
    )

    # 4. Build BM25
    tokenized = [doc.page_content.lower().split() for doc in chunks]

    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump({"documents": chunks, "bm25": bm25}, f)

    # 5. Update registry
    registry = _load_registry()
    coll_info = registry["collections"].get(collection_id, {})
    existing_files = set(coll_info.get("files", []))
    new_files = [fp.name for fp in file_paths]
    all_files = list(existing_files | set(new_files))

    coll_info["files"] = all_files
    coll_info["document_count"] = len(documents)
    coll_info["chunk_count"] = len(chunks)
    coll_info["status"] = "ready"
    registry["collections"][collection_id] = coll_info
    _save_registry(registry)

    return {
        "collection_id": collection_id,
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "files": all_files,
    }


def add_to_collection(
    collection_id: str,
    file_paths: List[Path],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Dict:
    """
    Add more files to an existing collection.

    Rebuilds the vector store and BM25 index with all documents
    (existing + new) for the collection.
    """
    coll_dir = COLLECTIONS_DIR / collection_id
    if not coll_dir.exists():
        return {"error": f"Collection '{collection_id}' does not exist."}

    # Get existing files from registry
    registry = _load_registry()
    coll_info = registry["collections"].get(collection_id, {})
    existing_files = coll_info.get("files", [])

    # Collect all file paths (existing + new)
    all_file_paths = []
    uploads_dir = coll_dir / "uploads"
    if uploads_dir.exists():
        for ext in SUPPORTED_EXTENSIONS:
            all_file_paths.extend(uploads_dir.glob(f"*{ext}"))

    all_file_paths.extend(file_paths)

    # Re-ingest everything
    return ingest_files(
        collection_id,
        all_file_paths,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
