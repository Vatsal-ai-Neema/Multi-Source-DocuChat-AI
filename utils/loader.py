"""
utils/loader.py
───────────────
Document Loading Module

Supports: PDF, TXT files
Returns: List of LangChain Document objects with metadata
"""

import os
from typing import List
from langchain_core.documents import Document


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return as list of Documents (one per page).
    Uses pdfplumber for better text extraction quality.
    """
    try:
        import pdfplumber

        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(Document(
                        page_content=text.strip(),
                        metadata={
                            "source": file_path,
                            "file_name": os.path.basename(file_path),
                            "page": page_num,
                            "total_pages": len(pdf.pages),
                            "file_type": "pdf"
                        }
                    ))
        return documents

    except ImportError:
        # Fallback: PyPDF2
        import PyPDF2
        documents = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(Document(
                        page_content=text.strip(),
                        metadata={
                            "source": file_path,
                            "file_name": os.path.basename(file_path),
                            "page": page_num,
                            "total_pages": len(reader.pages),
                            "file_type": "pdf"
                        }
                    ))
        return documents


def load_txt(file_path: str) -> List[Document]:
    """
    Load a plain text file.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if not content.strip():
        return []

    return [Document(
        page_content=content.strip(),
        metadata={
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "page": 1,
            "file_type": "txt"
        }
    )]


def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Main loader function.
    Accepts a list of file paths (PDF or TXT) and returns all documents.

    Args:
        file_paths: List of absolute/relative file paths

    Returns:
        List of LangChain Document objects
    """
    all_documents = []

    for path in file_paths:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue

        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            docs = load_pdf(path)
        elif ext == ".txt":
            docs = load_txt(path)
        else:
            print(f"[WARN] Unsupported file type: {ext} — skipping {path}")
            continue

        print(f"[INFO] Loaded '{os.path.basename(path)}': {len(docs)} page(s)")
        all_documents.extend(docs)

    print(f"[INFO] Total documents loaded: {len(all_documents)} page(s) across {len(file_paths)} file(s)")
    return all_documents
