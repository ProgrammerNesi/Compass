"""
Centralized configuration for the RAG application.

All configurable values live here. Override via environment variables
or by editing this file directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent

# ── LLM ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
# LangChain's ChatGoogleGenerativeAI expects GOOGLE_API_KEY
if GEMINI_API_KEY and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")

# ── Embeddings ───────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ── Reranker ─────────────────────────────────────────────────
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Retrieval ────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
MMR_FETCH_K = int(os.getenv("MMR_FETCH_K", "30"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))
RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "30"))
RRF_K = int(os.getenv("RRF_K", "60"))
MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "3"))

# ── Evaluation thresholds (runtime evaluator in agent) ───────
THRESHOLD_CONTEXT_PRECISION = float(os.getenv("THRESHOLD_CONTEXT_PRECISION", "0.70"))
THRESHOLD_CONTEXT_RECALL = float(os.getenv("THRESHOLD_CONTEXT_RECALL", "0.70"))
THRESHOLD_FAITHFULNESS = float(os.getenv("THRESHOLD_FAITHFULNESS", "0.80"))
THRESHOLD_ANSWER_RELEVANCY = float(os.getenv("THRESHOLD_ANSWER_RELEVANCY", "0.75"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))

# ── Paths ────────────────────────────────────────────────────
# Legacy baseline paths (used by benchmark scripts — DO NOT MOVE)
LEGACY_VECTOR_DB = BASE_DIR / "vector_db"
LEGACY_BM25_FILE = LEGACY_VECTOR_DB / "bm25.pkl"
KNOWLEDGE_BASE = BASE_DIR / "knowledge-base"

# New generalized paths
COLLECTIONS_DIR = BASE_DIR / "collections"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ── Agent ────────────────────────────────────────────────────
DEFAULT_MAX_ITERATIONS = int(os.getenv("DEFAULT_MAX_ITERATIONS", "5"))

# ── Supported file types ────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
