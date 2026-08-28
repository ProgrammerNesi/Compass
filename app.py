"""
RAG Agent — General-Purpose Document QA Application

A Gradio-based UI that allows users to upload documents,
build retrieval indexes, and ask questions using the existing
self-evaluating RAG agent.

Run:
    python app.py

Or with a specific port:
    python app.py --port 7860
"""

import sys
import json
import shutil
import argparse
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from config import (
    UPLOAD_DIR,
    COLLECTIONS_DIR,
    SUPPORTED_EXTENSIONS,
    DEFAULT_MAX_ITERATIONS,
)
from ingestion.collection_manager import (
    list_collections,
    get_collection_info,
    create_collection,
    ingest_files,
    delete_collection,
)
from retrieval.retriever import set_active, set_active_legacy, get_active
from agent.graph import graph


# ── Helper functions ─────────────────────────────────────────

def _get_collection_status():
    """Build a markdown summary of all collections."""
    collections = list_collections()
    if not collections:
        return "*No collections yet. Upload documents to create one.*"

    lines = ["| ID | Name | Files | Chunks | Status |",
             "|---|---|---|---|---|"]
    for c in collections:
        lines.append(
            f"| `{c['id']}` | {c.get('name', c['id'])} | "
            f"{c.get('document_count', 0)} | "
            f"{c.get('chunk_count', 0)} | "
            f"{c.get('status', 'unknown')} |"
        )
    return "\n".join(lines)


def _get_active_collection_label():
    active = get_active()
    if active is None:
        return "**Active collection:** None (using legacy baseline)"
    return f"**Active collection:** `{active.collection_id}`"


def _save_uploaded_files(files, collection_id):
    """Save uploaded files to the collection's upload directory."""
    coll_dir = COLLECTIONS_DIR / collection_id
    upload_dir = coll_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file in files:
        if file is None:
            continue
        # Gradio gives us a temp path; copy to our uploads dir
        src = Path(file.name) if hasattr(file, "name") else Path(file)
        dest = upload_dir / src.name
        shutil.copy2(str(src), str(dest))
        saved_paths.append(dest)

    return saved_paths


# ── UI Callback Functions ────────────────────────────────────

def on_create_collection(collection_name):
    """Create a new collection and activate it."""
    if not collection_name or not collection_name.strip():
        return (
            gr.update(),
            _get_collection_status(),
            "Please provide a collection name.",
        )

    cid = create_collection(name=collection_name.strip())
    set_active(cid)
    return (
        gr.update(value=cid),
        _get_collection_status(),
        f"Created and activated collection `{cid}`.",
    )


def on_activate_collection(collection_id):
    """Activate an existing collection."""
    if not collection_id or not collection_id.strip():
        return _get_active_collection_label(), "Please enter a collection ID."

    cid = collection_id.strip()
    info = get_collection_info(cid)
    if info is None:
        return _get_active_collection_label(), f"Collection `{cid}` not found."

    set_active(cid)
    return _get_active_collection_label(), f"Activated collection `{cid}`."


def on_upload_and_index(files, collection_id):
    """Upload files and build/update the collection indexes."""
    if not files:
        return (
            _get_collection_status(),
            _get_active_collection_label(),
            "No files provided.",
        )

    if not collection_id or not collection_id.strip():
        # Auto-create a collection
        collection_id = create_collection(name="uploaded_docs")

    cid = collection_id.strip()

    # Ensure collection exists
    info = get_collection_info(cid)
    if info is None:
        create_collection(cid)

    # Save files
    saved_paths = _save_uploaded_files(files, cid)

    if not saved_paths:
        return (
            _get_collection_status(),
            _get_active_collection_label(),
            "No valid files to process.",
        )

    # Ingest
    result = ingest_files(cid, saved_paths)

    # Activate
    set_active(cid)

    if "error" in result:
        return (
            _get_collection_status(),
            _get_active_collection_label(),
            f"Error: {result['error']}",
        )

    summary = (
        f"Processed {result['documents_loaded']} document(s) into "
        f"{result['chunks_created']} chunks. "
        f"Files: {', '.join(result['files'])}"
    )

    return (
        _get_collection_status(),
        _get_active_collection_label(),
        summary,
    )


def on_delete_collection(collection_id):
    """Delete a collection."""
    if not collection_id or not collection_id.strip():
        return _get_collection_status(), "Please enter a collection ID."

    cid = collection_id.strip()
    delete_collection(cid)

    active = get_active()
    if active and active.collection_id == cid:
        from retrieval.retriever import clear_active
        clear_active()

    return _get_collection_status(), f"Deleted collection `{cid}`."


def on_query(query, max_iterations, use_baseline):
    """Run the RAG agent on a query and return formatted results."""
    if not query or not query.strip():
        return "Please enter a query.", "", "", ""

    # Ensure correct collection is active
    if use_baseline:
        set_active_legacy()
    else:
        active = get_active()
        if active is None:
            return (
                "No collection active. Upload documents or activate a collection first.",
                "", "", "",
            )

    try:
        result = graph.invoke({
            "query": query.strip(),
            "max_iterations": int(max_iterations) if max_iterations else DEFAULT_MAX_ITERATIONS,
        })
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"**Error:** {str(e)}\n\n```\n{error_detail[-500:]}```", "", "", ""

    # Format answer
    answer = result.get("final_answer") or result.get("answer") or "No answer generated."
    confidence = result.get("confidence", "N/A")
    status = result.get("status", "unknown")
    iterations = result.get("iteration", 0)
    tools_used = result.get("tools_used", [])
    chunks = result.get("retrieved_chunks", [])

    # Format sources
    sources = []
    seen = set()
    for chunk in chunks:
        src = chunk.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            content_preview = chunk.get("content", "")[:200]
            sources.append(f"**{src}**\n> {content_preview}...")

    sources_text = "\n\n".join(sources) if sources else "*No sources retrieved.*"

    # Format metadata
    meta_parts = [
        f"**Status:** {status}",
        f"**Confidence:** {confidence}",
        f"**Iterations:** {iterations}",
        f"**Tools used:** {', '.join(tools_used) if tools_used else 'none'}",
        f"**Chunks retrieved:** {len(chunks)}",
    ]
    metadata_text = "\n\n".join(meta_parts)

    return answer, sources_text, metadata_text, ""


# ── Build the Gradio UI ─────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="RAG Agent — Document QA",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 📚 RAG Agent — Document QA")
        gr.Markdown(
            "Upload documents, build an index, and ask questions. "
            "The self-evaluating RAG agent retrieves, generates, "
            "evaluates, and retries until confident."
        )

        with gr.Tabs():
            # ── Tab 1: Collections ──────────────────────────
            with gr.Tab("📁 Collections"):
                gr.Markdown("### Create or activate a document collection")

                with gr.Row():
                    new_collection_name = gr.Textbox(
                        label="New collection name",
                        placeholder="e.g. my_project_docs",
                    )
                    create_btn = gr.Button("Create Collection", variant="primary")

                collection_id_input = gr.Textbox(
                    label="Collection ID to activate",
                    placeholder="e.g. abc12345",
                )
                activate_btn = gr.Button("Activate")
                delete_btn = gr.Button("Delete Collection", variant="stop")

                collection_status = gr.Markdown(
                    value=_get_collection_status(),
                    label="Collections",
                )
                collection_msg = gr.Markdown()

                create_btn.click(
                    on_create_collection,
                    inputs=[new_collection_name],
                    outputs=[collection_id_input, collection_status, collection_msg],
                )
                activate_btn.click(
                    on_activate_collection,
                    inputs=[collection_id_input],
                    outputs=[collection_status, collection_msg],
                )
                delete_btn.click(
                    on_delete_collection,
                    inputs=[collection_id_input],
                    outputs=[collection_status, collection_msg],
                )

            # ── Tab 2: Upload & Index ───────────────────────
            with gr.Tab("📤 Upload & Index"):
                gr.Markdown("### Upload documents to build or update an index")

                upload_collection_id = gr.Textbox(
                    label="Collection ID (leave blank to auto-create)",
                    placeholder="e.g. abc12345",
                )

                file_upload = gr.File(
                    label="Upload documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md", ".docx"],
                )

                upload_btn = gr.Button("Process & Index", variant="primary")

                active_label = gr.Markdown(
                    value=_get_active_collection_label(),
                )
                upload_result = gr.Markdown()

                upload_btn.click(
                    on_upload_and_index,
                    inputs=[file_upload, upload_collection_id],
                    outputs=[collection_status, active_label, upload_result],
                )

            # ── Tab 3: Query ────────────────────────────────
            with gr.Tab("💬 Ask Questions"):
                gr.Markdown("### Ask a question about your documents")

                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your question",
                        placeholder="e.g. What are the main findings?",
                        lines=2,
                    )

                with gr.Row():
                    max_iter_input = gr.Slider(
                        label="Max iterations",
                        minimum=1,
                        maximum=10,
                        value=DEFAULT_MAX_ITERATIONS,
                        step=1,
                    )
                    use_baseline = gr.Checkbox(
                        label="Use legacy baseline collection",
                        value=False,
                    )

                query_btn = gr.Button("Ask", variant="primary")

                with gr.Row():
                    with gr.Column():
                        answer_output = gr.Markdown(label="Answer")
                    with gr.Column():
                        sources_output = gr.Markdown(label="Sources")
                        metadata_output = gr.Markdown(label="Agent Info")

                query_btn.click(
                    on_query,
                    inputs=[query_input, max_iter_input, use_baseline],
                    outputs=[answer_output, sources_output, metadata_output, upload_result],
                )

        return demo


# ── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Agent UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
