"""
utils/embedder.py
─────────────────
Embedding & Vector Store Module

Converts text chunks → dense vectors → stores in FAISS index.

Embedding Model: HuggingFace 'all-MiniLM-L6-v2'
  → Free, runs locally (no API key needed)
  → 384-dimensional vectors
  → Great speed/quality tradeoff for semantic search

Vector Store: FAISS (Facebook AI Similarity Search)
  → Stores all vectors in memory (fast)
  → Supports cosine similarity search
  → Can be saved/loaded from disk
"""

import os
import pickle
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ── Configuration ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_DIR = "faiss_index"   # Local directory to save/load the index
LOCAL_CACHE_DIR = ".model_cache"

# ─────────────────────────────────────────────────────────────────────────────


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace embedding model.
    First call downloads the model (~90MB), subsequent calls use cache.
    """
    cache_dir = os.path.abspath(LOCAL_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)

    # Keep model downloads inside the project to avoid global cache permission issues.
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(cache_dir, "sentence_transformers")

    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"],
        model_kwargs={"device": "cpu"},          # Use GPU if available: "cuda"
        encode_kwargs={"normalize_embeddings": True}  # Normalize for cosine sim
    )
    return embeddings


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Build a FAISS vector store from document chunks.

    Steps:
      1. Embed each chunk using HuggingFace model
      2. Store vectors + text in FAISS index
      3. Save index to disk (for future use)

    Args:
        chunks: List of Document chunks from chunker

    Returns:
        FAISS vector store object (in-memory + saved to disk)
    """
    print(f"[INFO] Building FAISS index from {len(chunks)} chunks...")

    embeddings = get_embedding_model()

    # Create FAISS index
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save to disk for later use
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"[INFO] FAISS index saved to '{FAISS_INDEX_DIR}/'")

    return vectorstore


def load_vectorstore() -> Optional[FAISS]:
    """
    Load a previously saved FAISS index from disk.
    Returns None if no saved index exists.
    """
    if not os.path.exists(FAISS_INDEX_DIR):
        print("[WARN] No saved FAISS index found.")
        return None

    print(f"[INFO] Loading FAISS index from '{FAISS_INDEX_DIR}/'")
    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True  # Safe for local use
    )
    return vectorstore
