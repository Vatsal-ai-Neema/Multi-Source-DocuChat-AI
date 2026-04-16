"""
utils/retriever.py
──────────────────
Retrieval Module — The Heart of RAG
Version 2.0 — HyDE Integrated

Original RAG flow:
  User question → embed → FAISS search → chunks

Upgraded RAG flow with HyDE (Upgrade 04):
  User question
    → Gemini imagines what a perfect answer doc looks like
    → Embed THAT hypothetical document
    → FAISS search finds chunks closest to ideal answer
  Result: 35-40% better retrieval accuracy (Gao et al., 2022)

New additions:
  - retrieve_with_hyde()  — HyDE-powered retrieval [Upgrade 04]
  - get_chunks_by_doc()   — groups chunks per document [used by Upgrade 07]
  - build_context_string() — unchanged, used everywhere
"""

from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


# ══════════════════════════════════════════════════════════════════════════════
# ORIGINAL — Standard Retrieval (unchanged, used as fallback)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_relevant_chunks(
    query: str,
    vectorstore: FAISS,
    k: int = 4,
    score_threshold: float = 0.3
) -> List[Document]:
    """
    Standard RAG retrieval — embed query, search FAISS, return top-K chunks.

    Args:
        query:           User's raw question string
        vectorstore:     FAISS index built from uploaded documents
        k:               Number of chunks to retrieve
        score_threshold: Minimum cosine similarity score (0-1)

    Returns:
        List of relevant Document chunks with metadata
    """
    if vectorstore is None:
        raise ValueError("Vector store is not initialized. Please process documents first.")

    results_with_scores: List[Tuple[Document, float]] = (
        vectorstore.similarity_search_with_score(query, k=k)
    )

    filtered = [
        doc for doc, score in results_with_scores
        if score >= score_threshold
    ]

    if not filtered and results_with_scores:
        filtered = [results_with_scores[0][0]]

    for i, (doc, score) in enumerate(results_with_scores[:k]):
        source = doc.metadata.get("file_name", "Unknown")
        page   = doc.metadata.get("page", "?")
        print(f"[RETRIEVAL] Chunk {i+1}: score={score:.3f} | {source} (p.{page})")

    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# UPGRADE 04 — HyDE: Hypothetical Document Embedder
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_with_hyde(
    question: str,
    vectorstore: FAISS,
    k: int = 4,
    score_threshold: float = 0.25
) -> Tuple[List[Document], str]:
    """
    [UPGRADE 04 — HyDE Retrieval]

    Instead of embedding the raw user question, we:
      1. Ask Gemini to write a hypothetical ideal-answer paragraph
      2. Embed THAT paragraph (much closer in vector space to real doc chunks)
      3. Use that embedding to search FAISS

    Why this is better:
      - User: "what is photosynthesis?" => short, generic embedding
      - HyDE: "Photosynthesis is the process by which plants convert
               sunlight into glucose using chlorophyll..." => rich embedding
      - Rich embedding finds relevant chunks much more accurately

    Falls back to standard retrieval if HyDE generation fails.

    Args:
        question:        Original user question
        vectorstore:     FAISS index
        k:               Number of chunks to retrieve
        score_threshold: Minimum similarity score

    Returns:
        Tuple of (retrieved chunks list, hypothetical_doc_used_for_search)
    """
    from utils.llm import generate_hypothetical_document

    # Step 1: Generate hypothetical document
    hypothetical_doc = generate_hypothetical_document(question)

    # Step 2: Search using hypothetical doc as query
    try:
        results_with_scores = vectorstore.similarity_search_with_score(
            hypothetical_doc, k=k
        )

        filtered = [
            doc for doc, score in results_with_scores
            if score >= score_threshold
        ]

        if not filtered and results_with_scores:
            filtered = [results_with_scores[0][0]]

        for i, (doc, score) in enumerate(results_with_scores[:k]):
            source = doc.metadata.get("file_name", "Unknown")
            page   = doc.metadata.get("page", "?")
            print(f"[HyDE RETRIEVAL] Chunk {i+1}: score={score:.3f} | {source} (p.{page})")

        return filtered, hypothetical_doc

    except Exception as e:
        print(f"[HyDE] Search failed, falling back to standard: {e}")
        chunks = retrieve_relevant_chunks(question, vectorstore, k, score_threshold)
        return chunks, question


# ══════════════════════════════════════════════════════════════════════════════
# UPGRADE 07 Helper — Group chunks by document name
# ══════════════════════════════════════════════════════════════════════════════

def get_chunks_by_doc(
    vectorstore: FAISS,
    query: str = "main content summary overview",
    k_per_doc: int = 3
) -> Dict[str, List[str]]:
    """
    [Helper for Upgrade 07 — Contradiction Detector]

    Retrieves chunks and groups them by source document name.
    Used by detect_contradictions() to compare text across documents.

    Args:
        vectorstore:  FAISS index
        query:        Broad query to fetch representative chunks
        k_per_doc:    Max chunks to keep per document

    Returns:
        Dict of {document_name: [chunk_text, chunk_text, ...]}
    """
    try:
        results = vectorstore.similarity_search(query, k=30)
    except Exception:
        return {}

    chunks_by_doc: Dict[str, List[str]] = {}

    for doc in results:
        name = doc.metadata.get("file_name", "Unknown")
        if name not in chunks_by_doc:
            chunks_by_doc[name] = []
        if len(chunks_by_doc[name]) < k_per_doc:
            chunks_by_doc[name].append(doc.page_content.strip())

    print(f"[CHUNKS BY DOC] Found {len(chunks_by_doc)} documents: {list(chunks_by_doc.keys())}")
    return chunks_by_doc


# ══════════════════════════════════════════════════════════════════════════════
# ORIGINAL — Unchanged helper functions
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_with_scores(
    query: str,
    vectorstore: FAISS,
    k: int = 4
) -> List[Tuple[Document, float]]:
    """Returns chunks with their similarity scores."""
    return vectorstore.similarity_search_with_score(query, k=k)


def build_context_string(chunks: List[Document]) -> str:
    """
    Format retrieved chunks into a clean context string for the LLM.
    Adds source citation markers so the LLM can reference them.
    """
    if not chunks:
        return ""

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source  = chunk.metadata.get("file_name", "Unknown")
        page    = chunk.metadata.get("page", "?")
        content = chunk.page_content.strip()
        context_parts.append(f"[Source {i}: {source}, Page {page}]\n{content}")

    return "\n\n---\n\n".join(context_parts)
