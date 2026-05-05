"""
ingest.py — RTI Document Ingestion Pipeline
============================================
Loads official RTI PDFs, splits them into semantically-aware chunks,
generates embeddings using sentence-transformers/all-MiniLM-L6-v2 (locally,
on CPU), and persists the vector store to ChromaDB.

ML Best Practices Applied:
- Semantic, section-aware chunking (smaller chunks → better recall precision)
- Deterministic chunk IDs (sha256 hash) → idempotent re-ingestion, no duplicates
- Per-document `doc_type` metadata → enables future filtered retrieval
- Overlap keeps context across boundaries without inflating chunk count
"""

import os
import hashlib
import re
import shutil
import argparse
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ──────────────────────────── Configuration ────────────────────────────────
DATA_PATH    = "data"
CHROMA_PATH  = "chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"

# Chunk size: 700 chars is sweet-spot for MiniLM (512-token window).
# Smaller than 1000 → finer-grained retrieval for specific facts (fees, dates).
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 150

# RTI-specific separators: preserve legal section/chapter boundaries first,
# then fall back to paragraph → sentence → character splits.
RTI_SEPARATORS = [
    # Legal section headers
    r"\n(?=Section\s+\d+)",
    r"\n(?=SECTION\s+\d+)",
    r"\n(?=Chapter\s+[IVX]+)",
    r"\n(?=CHAPTER\s+[IVX]+)",
    r"\n(?=Rule\s+\d+)",
    r"\n(?=RULE\s+\d+)",
    # Paragraph and sentence boundaries
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

# Map source filename substrings → human-readable document type tag.
# Used as metadata for future filtered retrieval.
DOC_TYPE_MAP = {
    "faq":     "FAQ",
    "manual":  "Manual",
    "rules":   "Rules",
    "act":     "Act",
}


# ───────────────────────────── Helpers ─────────────────────────────────────

def classify_doc_type(source_path: str) -> str:
    """Return a doc_type label based on the filename."""
    name = os.path.basename(source_path).lower()
    for key, label in DOC_TYPE_MAP.items():
        if key in name:
            return label
    return "Other"


def make_chunk_id(doc: Document) -> str:
    """
    Generate a deterministic 12-char chunk ID:
        sha256(source_path + page_number + start_index)
    Enables idempotent re-ingestion — re-running ingest.py won't duplicate
    chunks that already exist in ChromaDB.
    """
    source = doc.metadata.get("source", "")
    page   = str(doc.metadata.get("page", ""))
    start  = str(doc.metadata.get("start_index", ""))
    raw    = f"{source}::{page}::{start}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def enrich_metadata(docs: List[Document]) -> List[Document]:
    """Add doc_type, chunk_id to each document's metadata."""
    for doc in docs:
        doc.metadata["doc_type"]  = classify_doc_type(doc.metadata.get("source", ""))
        doc.metadata["chunk_id"]  = make_chunk_id(doc)
    return docs


# ─────────────────────────── Main Pipeline ─────────────────────────────────

def ingest_documents(reset: bool = False) -> None:
    """
    Full ingestion pipeline:
      1. Load PDFs
      2. Split into semantic chunks
      3. Enrich metadata (doc_type, chunk_id)
      4. Deduplicate against existing ChromaDB (if reset=False)
      5. Embed + persist
    """

    # ── Optionally wipe the vector store ──────────────────────────────────
    if reset and os.path.exists(CHROMA_PATH):
        print(f"[RESET] Wiping existing vector store at '{CHROMA_PATH}'...")
        shutil.rmtree(CHROMA_PATH)

    # ── Step 1: Load PDFs ─────────────────────────────────────────────────
    print(f"\n[1/4] Loading PDFs from '{DATA_PATH}'...")
    if not os.path.isdir(DATA_PATH):
        print(f"  ERROR: Data directory '{DATA_PATH}' not found.")
        return

    loader    = PyPDFDirectoryLoader(DATA_PATH)
    raw_docs  = loader.load()

    if not raw_docs:
        print("  ERROR: No PDF documents found.")
        return

    # Count pages per file for summary
    per_file: dict = {}
    for d in raw_docs:
        fn = os.path.basename(d.metadata.get("source", "unknown"))
        per_file[fn] = per_file.get(fn, 0) + 1

    print(f"  Loaded {len(raw_docs)} pages from {len(per_file)} PDF(s):")
    for fn, pages in per_file.items():
        print(f"    • {fn}  ({pages} pages)")

    # ── Step 2: Smart Chunking ────────────────────────────────────────────
    print(f"\n[2/4] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        separators=RTI_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=True,   # regex separators for section headers
        add_start_index=True,      # enables precise offset tracking
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  Created {len(chunks)} chunks  (avg {sum(len(c.page_content) for c in chunks)//len(chunks)} chars/chunk)")

    # ── Step 3: Enrich Metadata ───────────────────────────────────────────
    print("\n[3/4] Enriching metadata (doc_type + chunk_id)...")
    chunks = enrich_metadata(chunks)

    # Deduplication: load existing IDs from ChromaDB and skip already-ingested chunks
    existing_ids: set = set()
    if os.path.exists(CHROMA_PATH) and not reset:
        print("  Checking for existing chunks in ChromaDB (idempotent mode)...")
        _tmp_embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _tmp_db    = Chroma(persist_directory=CHROMA_PATH, embedding_function=_tmp_embed)
        existing   = _tmp_db.get()
        existing_ids = set(existing.get("ids", []))
        print(f"  Found {len(existing_ids)} existing chunk IDs.")

    new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]
    skipped    = len(chunks) - len(new_chunks)
    if skipped:
        print(f"  Skipping {skipped} already-ingested chunks. Adding {len(new_chunks)} new chunks.")
    else:
        print(f"  All {len(new_chunks)} chunks are new.")

    if not new_chunks:
        print("\n  Nothing new to ingest. Vector store is up-to-date.")
        return

    # ── Step 4: Embed + Persist ───────────────────────────────────────────
    print(f"\n[4/4] Embedding {len(new_chunks)} chunks with '{EMBED_MODEL}' (CPU)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    ids = [c.metadata["chunk_id"] for c in new_chunks]

    if os.path.exists(CHROMA_PATH) and not reset:
        # Append to existing store
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        db.add_documents(new_chunks, ids=ids)
    else:
        # Create fresh store (Chroma 0.4.x auto-persists; no .persist() call needed)
        db = Chroma.from_documents(
            documents=new_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            ids=ids,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    total = db._collection.count()
    print(f"\n[DONE] Ingestion complete.")
    print(f"   Vector store: '{CHROMA_PATH}'  |  Total chunks: {total}")

    # Doc-type breakdown
    dt_counts: dict = {}
    for c in chunks:
        dt = c.metadata.get("doc_type", "Other")
        dt_counts[dt] = dt_counts.get(dt, 0) + 1
    print("   Chunk breakdown by document type:")
    for dt, cnt in sorted(dt_counts.items()):
        print(f"    - {dt:<10}: {cnt}")


# ───────────────────────────── Entry Point ─────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest RTI PDFs into ChromaDB vector store."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing ChromaDB and re-ingest from scratch.",
    )
    args = parser.parse_args()
    ingest_documents(reset=args.reset)
