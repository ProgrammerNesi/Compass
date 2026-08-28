"""
Multi-format document loaders.

Supports: PDF, TXT, Markdown, DOCX.
Each loader returns a list of LangChain Document objects with
standardized metadata (source, doc_type, page_number when available).

To add a new format, create a loader function following the same
signature: load_<ext>(file_path: Path) -> List[Document]
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_pdf(file_path: Path) -> List[Document]:
    """Load a PDF file using PyPDFLoader."""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_type"] = "pdf"
        doc.metadata["source"] = file_path.name
    return docs


def load_txt(file_path: Path) -> List[Document]:
    """Load a plain text file."""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_type"] = "txt"
        doc.metadata["source"] = file_path.name
    return docs


def load_md(file_path: Path) -> List[Document]:
    """Load a Markdown file."""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_type"] = "md"
        doc.metadata["source"] = file_path.name
    return docs


def load_docx(file_path: Path) -> List[Document]:
    """Load a DOCX file using Docx2txtLoader."""
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(str(file_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_type"] = "docx"
        doc.metadata["source"] = file_path.name
    return docs


# Registry: extension -> loader function
LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md": load_md,
    ".docx": load_docx,
}


def load_file(file_path: Path) -> List[Document]:
    """
    Load a single file using the appropriate loader.

    Returns a list of Documents with metadata:
      - source: filename
      - doc_type: file extension type
      - page_number: (PDF only) page number
    """
    ext = file_path.suffix.lower()

    if ext not in LOADERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(LOADERS.keys()))}"
        )

    return LOADERS[ext](file_path)


def load_files(file_paths: List[Path]) -> List[Document]:
    """
    Load multiple files, collecting all documents.

    Returns a flat list of all Documents from all files.
    """
    all_docs = []
    errors = []

    for fp in file_paths:
        try:
            docs = load_file(fp)
            all_docs.extend(docs)
        except Exception as e:
            errors.append({"file": fp.name, "error": str(e)})

    if errors:
        import warnings
        warnings.warn(
            f"Failed to load {len(errors)} file(s): "
            + "; ".join(f"{e['file']}: {e['error']}" for e in errors)
        )

    return all_docs
