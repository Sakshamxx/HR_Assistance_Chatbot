#!/usr/bin/env python3
"""
ingest.py - Build the vector knowledge base for the HR policy assistant.

Pipeline:
    knowledge_base/*  ->  type-aware loaders
                      ->  structure split (markdown headers | PDF pages)
                      ->  recursive character split (size-capped chunks)
                      ->  Gemini embeddings (RETRIEVAL_DOCUMENT)
                      ->  persistent Chroma store at ./chroma_db

Run with the venv interpreter:  venv/bin/python ingest.py
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = Path("knowledge_base")
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "hr_policies"

# gemini-embedding-001 is Google's current GA text embedding model.
# (text-embedding-004 and embedding-001 are both deprecated - do not use.)
EMBEDDING_MODEL = "gemini-embedding-001"

# Final chunk ceiling: small enough for the embedding model and for stuffing
# several chunks into the LLM context window, large enough to stay coherent.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Extension -> loader factory. Anything not listed here is skipped with a warning.
LOADERS = {
    ".pdf": lambda p: PyPDFLoader(str(p)),
    ".docx": lambda p: Docx2txtLoader(str(p)),
    ".md": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
}

# A line that starts with "##" followed by text = a real Markdown H2 heading.
MD_HEADER_RE = re.compile(r"^##\s+\S", re.MULTILINE)

# Header levels fed to MarkdownHeaderTextSplitter, coarsest first.
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


# ---------------------------------------------------------------------------
# Step 1b: load every document in knowledge_base/, auto-detecting its type
# from the file extension. Unsupported types and unreadable files produce a
# warning and are skipped, so one bad file can never abort the whole ingest.
# Returns {file path: [Documents]} so later steps can reason per source file.
# ---------------------------------------------------------------------------
def load_documents(kb_dir: Path) -> dict[Path, list[Document]]:
    if not kb_dir.is_dir():
        sys.exit(f"ERROR: '{kb_dir}/' does not exist. Create it and add policy documents.")

    files = sorted(
        p for p in kb_dir.rglob("*") if p.is_file() and not p.name.startswith(".")
    )
    if not files:
        sys.exit(
            f"ERROR: '{kb_dir}/' is empty. Add at least one .pdf, .docx, .md or .txt "
            "policy document before running ingest."
        )

    loaded: dict[Path, list[Document]] = {}
    for path in files:
        extension = path.suffix.lower()
        loader_factory = LOADERS.get(extension)

        if loader_factory is None:
            print(f"  WARNING  unsupported file type '{extension or '(none)'}' - skipping {path}")
            continue

        try:
            docs = loader_factory(path).load()
        except Exception as exc:  # corrupt/encrypted/locked file: warn, keep going
            print(f"  WARNING  could not load {path}: {type(exc).__name__}: {exc}")
            continue

        if not any(d.page_content.strip() for d in docs):
            print(f"  WARNING  no extractable text in {path} (scanned image PDF?) - skipping")
            continue

        # Normalise the source path so every chunk can cite its origin file.
        for d in docs:
            d.metadata["source"] = str(path)

        loaded[path] = docs
        print(f"  loaded   {path}  ({len(docs)} page/doc object(s))")

    return loaded


# ---------------------------------------------------------------------------
# Step 2: hybrid structure-aware split.
#   - Document HAS "##" Markdown headers -> MarkdownHeaderTextSplitter, and the
#     section name travels with the chunk under the 'section' metadata key.
#   - Document has NO "##" headers (a plain PDF, say) -> keep one chunk per page
#     and preserve the page number PyPDFLoader already attached, under 'page'.
# ---------------------------------------------------------------------------
def split_by_structure(path: Path, docs: list[Document]) -> list[Document]:
    full_text = "\n\n".join(d.page_content for d in docs)

    if MD_HEADER_RE.search(full_text):
        # strip_headers=False keeps the heading text inside the chunk body, which
        # measurably helps retrieval ("## Parental Leave" is a strong signal).
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        sections = splitter.split_text(full_text)

        structured: list[Document] = []
        for section in sections:
            # Collect whichever header levels this chunk actually sits under.
            breadcrumbs = [
                section.metadata[key]
                for _, key in HEADERS_TO_SPLIT_ON
                if section.metadata.get(key)
            ]
            section.metadata = {
                "source": str(path),
                "split_by": "markdown_header",
                # 'section' = the most specific heading above this chunk.
                "section": breadcrumbs[-1] if breadcrumbs else "(untitled section)",
                # Full heading trail, useful for citations in the chat UI.
                "section_path": " > ".join(breadcrumbs) if breadcrumbs else "(untitled section)",
            }
            structured.append(section)

        print(f"  split    {path}  -> {len(structured)} section(s) by Markdown headers")
        return structured

    # No Markdown headers: fall back to page-level structure.
    structured = []
    for index, doc in enumerate(docs):
        metadata = {
            "source": str(path),
            "split_by": "page",
            # PyPDFLoader's own 'page' value (0-indexed); index is the fallback
            # for loaders that do not paginate, e.g. Docx2txtLoader.
            "page": doc.metadata.get("page", index),
        }
        page_label = doc.metadata.get("page_label")
        if page_label is not None:
            metadata["page_label"] = str(page_label)  # the printed page number
        structured.append(Document(page_content=doc.page_content, metadata=metadata))

    print(f"  split    {path}  -> {len(structured)} page chunk(s), page numbers kept")
    return structured


# ---------------------------------------------------------------------------
# Step 3: cap chunk size. RecursiveCharacterTextSplitter.split_documents()
# copies each parent chunk's metadata onto every child, so the 'section' and
# 'page' keys set in step 2 survive this second pass untouched.
# ---------------------------------------------------------------------------
def split_by_size(structured_chunks: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # records each chunk's offset in its parent
    )
    return splitter.split_documents(structured_chunks)


# ---------------------------------------------------------------------------
# Chroma rejects None metadata values, so drop empty keys before storing.
# ---------------------------------------------------------------------------
def clean_metadata(chunks: list[Document]) -> list[Document]:
    for chunk in chunks:
        chunk.metadata = {
            k: v
            for k, v in chunk.metadata.items()
            if isinstance(v, (str, int, float, bool))
        }
    return chunks


def main() -> None:
    # -----------------------------------------------------------------------
    # Step 1a: load API keys from .env into the process environment.
    # GoogleGenerativeAIEmbeddings picks up GOOGLE_API_KEY from there.
    # -----------------------------------------------------------------------
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        # Accept GEMINI_API_KEY as an alias, since both names are in circulation.
        if os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        else:
            sys.exit(
                "ERROR: GOOGLE_API_KEY is not set.\n"
                "       Add this line to .env:  GOOGLE_API_KEY=your_key_here\n"
                "       Get a key at https://aistudio.google.com/apikey"
            )

    print(f"\nReading documents from {KNOWLEDGE_BASE_DIR}/ ...")
    documents_by_file = load_documents(KNOWLEDGE_BASE_DIR)
    if not documents_by_file:
        sys.exit("ERROR: no supported documents could be loaded. Nothing to ingest.")

    # -----------------------------------------------------------------------
    # Steps 2 + 3: structure split per file, then size-cap every chunk.
    # -----------------------------------------------------------------------
    print("\nSplitting documents ...")
    structured_chunks: list[Document] = []
    for path, docs in documents_by_file.items():
        structured_chunks.extend(split_by_structure(path, docs))

    chunks = clean_metadata(split_by_size(structured_chunks))
    print(
        f"\n  {len(structured_chunks)} structural chunk(s) -> {len(chunks)} final chunk(s) "
        f"(max {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap)"
    )

    # -----------------------------------------------------------------------
    # Step 4: embeddings. task_type="RETRIEVAL_DOCUMENT" tells Google these are
    # corpus documents being indexed for later search, not the search query
    # itself - the query side uses RETRIEVAL_QUERY in app.py.
    # -----------------------------------------------------------------------
    print(f"\nEmbedding {len(chunks)} chunk(s) with {EMBEDDING_MODEL} ...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
    )

    # -----------------------------------------------------------------------
    # Step 5: persist chunks + embeddings + metadata to Chroma at ./chroma_db.
    # The old store is removed first so re-running never duplicates chunks.
    # -----------------------------------------------------------------------
    if PERSIST_DIR.exists():
        print(f"  removing previous vector store at {PERSIST_DIR}/ (full rebuild)")
        shutil.rmtree(PERSIST_DIR)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )

    # -----------------------------------------------------------------------
    # Summary: source count, chunk count, and sample metadata to eyeball.
    # -----------------------------------------------------------------------
    stored = vector_store._collection.count()
    print("\n" + "=" * 68)
    print("INGEST COMPLETE")
    print("=" * 68)
    print(f"  Source documents loaded : {len(documents_by_file)}")
    for path in documents_by_file:
        print(f"      - {path}")
    print(f"  Chunks created          : {len(chunks)}")
    print(f"  Vectors stored in Chroma: {stored}")
    print(f"  Vector store location   : {PERSIST_DIR.resolve()}")
    print(f"  Collection name         : {COLLECTION_NAME}")

    # One example per splitting strategy that actually ran.
    print("\n  Example chunk metadata:")
    for strategy in ("markdown_header", "page"):
        example = next((c for c in chunks if c.metadata.get("split_by") == strategy), None)
        if example is None:
            continue
        print(f"\n    [{strategy}]")
        print(f"      metadata : {example.metadata}")
        preview = " ".join(example.page_content.split())[:160]
        print(f"      content  : {preview}{'...' if len(example.page_content) > 160 else ''}")
    print()


if __name__ == "__main__":
    main()
