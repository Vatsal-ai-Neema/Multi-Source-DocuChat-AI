"""
utils/chunker.py
────────────────
Text Chunking Module

Splits large documents into smaller overlapping chunks.
This is critical for RAG — too large = noisy context,
too small = missing context.

Strategy: RecursiveCharacterTextSplitter
  → Tries to split on: paragraphs → sentences → words → characters
  → Maintains semantic coherence as much as possible
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Configuration ────────────────────────────────────────────────────────────

CHUNK_SIZE    = 800   # Max characters per chunk (~150-200 words)
CHUNK_OVERLAP = 150   # Overlap between chunks (preserves context at boundaries)

# ─────────────────────────────────────────────────────────────────────────────


def split_into_chunks(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Split documents into smaller, overlapping text chunks.

    Args:
        documents:     List of LangChain Document objects (from loader)
        chunk_size:    Max characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of smaller Document chunks with preserved metadata
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Split priority: paragraph > sentence > word > character
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # ── Enrich metadata ───────────────────────────────────────────────────────
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"[INFO] Chunking complete: {len(documents)} doc(s) → {len(chunks)} chunks")
    print(f"[INFO] Avg chunk size: {sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} chars")

    return chunks


def preview_chunks(chunks: List[Document], n: int = 3) -> None:
    """
    Debug helper: Print first N chunks to inspect splitting quality.
    """
    print(f"\n{'='*60}")
    print(f"CHUNK PREVIEW (first {n} of {len(chunks)})")
    print('='*60)
    for i, chunk in enumerate(chunks[:n]):
        print(f"\n[Chunk {i+1}]")
        print(f"Source  : {chunk.metadata.get('file_name', 'N/A')}")
        print(f"Page    : {chunk.metadata.get('page', 'N/A')}")
        print(f"Length  : {len(chunk.page_content)} chars")
        print(f"Content : {chunk.page_content[:200]}...")
        print('-'*60)
