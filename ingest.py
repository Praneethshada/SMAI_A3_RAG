import os
import hashlib
import shutil
import argparse
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_PATH    = "data"
CHROMA_PATH  = "chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 150

# Section-aware separators: preserve RTI legal boundaries before falling
# back to paragraph, sentence, and character splits.
RTI_SEPARATORS = [
    r"\n(?=Section\s+\d+)",
    r"\n(?=SECTION\s+\d+)",
    r"\n(?=Chapter\s+[IVX]+)",
    r"\n(?=CHAPTER\s+[IVX]+)",
    r"\n(?=Rule\s+\d+)",
    r"\n(?=RULE\s+\d+)",
    "\n\n", "\n", ". ", " ", "",
]

DOC_TYPE_MAP = {
    "faq": "FAQ", "manual": "Manual", "rules": "Rules", "act": "Act",
}


def classify_doc_type(source_path: str) -> str:
    name = os.path.basename(source_path).lower()
    for key, label in DOC_TYPE_MAP.items():
        if key in name:
            return label
    return "Other"


def make_chunk_id(doc: Document) -> str:
    """Deterministic 12-char ID: sha256(source + page + start_index)."""
    raw = f"{doc.metadata.get('source', '')}::{doc.metadata.get('page', '')}::{doc.metadata.get('start_index', '')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def enrich_metadata(docs: List[Document]) -> List[Document]:
    for doc in docs:
        doc.metadata["doc_type"] = classify_doc_type(doc.metadata.get("source", ""))
        doc.metadata["chunk_id"] = make_chunk_id(doc)
    return docs


def ingest_documents(reset: bool = False) -> None:
    if reset and os.path.exists(CHROMA_PATH):
        print(f"[RESET] Wiping '{CHROMA_PATH}'...")
        shutil.rmtree(CHROMA_PATH)

    print(f"\n[1/4] Loading PDFs from '{DATA_PATH}'...")
    if not os.path.isdir(DATA_PATH):
        print(f"  ERROR: '{DATA_PATH}' not found.")
        return

    raw_docs = PyPDFDirectoryLoader(DATA_PATH).load()
    if not raw_docs:
        print("  ERROR: No PDF documents found.")
        return

    per_file: dict = {}
    for d in raw_docs:
        fn = os.path.basename(d.metadata.get("source", "unknown"))
        per_file[fn] = per_file.get(fn, 0) + 1

    print(f"  Loaded {len(raw_docs)} pages from {len(per_file)} PDF(s):")
    for fn, pages in per_file.items():
        print(f"    {fn}  ({pages} pages)")

    print(f"\n[2/4] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        separators=RTI_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=True,
        add_start_index=True,
    )
    chunks = splitter.split_documents(raw_docs)
    avg = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"  Created {len(chunks)} chunks  (avg {avg} chars/chunk)")

    print("\n[3/4] Enriching metadata (doc_type + chunk_id)...")
    chunks = enrich_metadata(chunks)

    existing_ids: set = set()
    if os.path.exists(CHROMA_PATH) and not reset:
        print("  Checking for existing chunks (idempotent mode)...")
        _tmp = Chroma(persist_directory=CHROMA_PATH,
                      embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL))
        existing_ids = set(_tmp.get().get("ids", []))
        print(f"  Found {len(existing_ids)} existing chunk IDs.")

    new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]
    skipped = len(chunks) - len(new_chunks)
    if skipped:
        print(f"  Skipping {skipped} duplicates. Adding {len(new_chunks)} new chunks.")
    else:
        print(f"  All {len(new_chunks)} chunks are new.")

    if not new_chunks:
        print("\n  Nothing new to ingest. Vector store is up-to-date.")
        return

    print(f"\n[4/4] Embedding {len(new_chunks)} chunks with '{EMBED_MODEL}' (CPU)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    ids = [c.metadata["chunk_id"] for c in new_chunks]

    if os.path.exists(CHROMA_PATH) and not reset:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        db.add_documents(new_chunks, ids=ids)
    else:
        db = Chroma.from_documents(
            documents=new_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            ids=ids,
        )

    total = db._collection.count()
    print(f"\n[DONE] Ingestion complete.")
    print(f"   Vector store: '{CHROMA_PATH}'  |  Total chunks: {total}")

    dt_counts: dict = {}
    for c in chunks:
        dt = c.metadata.get("doc_type", "Other")
        dt_counts[dt] = dt_counts.get(dt, 0) + 1
    print("   Chunk breakdown by document type:")
    for dt, cnt in sorted(dt_counts.items()):
        print(f"    - {dt:<10}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest RTI PDFs into ChromaDB.")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe existing ChromaDB and re-ingest from scratch.")
    ingest_documents(reset=parser.parse_args().reset)
