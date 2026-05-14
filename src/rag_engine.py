"""
src/rag_engine.py
=================
Retrieval-Augmented Generation (RAG) engine for scientific paper search.

WHY THIS FILE EXISTS:
  The original script ran top-to-bottom as a one-off. Refactored into clean,
  composable functions so any consumer (Streamlit, FastAPI, CLI) can call
  load_pdf() → create_chunks() → create_embeddings() → search() independently.

HOW RAG WORKS (for interviews):
  1. INDEXING  — Parse PDF text → split into overlapping chunks →
                 embed with SentenceTransformer → store in FAISS index.
  2. RETRIEVAL — Embed the user's query → find nearest chunks in FAISS →
                 return top-k most semantically relevant passages.
  3. GENERATION (optional, requires LLM) — Pass retrieved chunks as context
                 to an LLM (e.g. Claude, GPT-4) to generate an answer.

INTERVIEW VALUE:
  Understanding the distinction between dense retrieval (FAISS + embeddings)
  and sparse retrieval (BM25/TF-IDF) is a common interview question. This
  implementation uses dense retrieval with cosine similarity via FAISS.
"""

import os
import re
import numpy as np
from typing import List, Tuple, Optional

# Default paths
PAPERS_DIR  = os.path.join(os.path.dirname(__file__), "..", "papers")
DEFAULT_PDF = os.path.join(PAPERS_DIR, "sample.pdf")

# Embedding model name — small, fast, good quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 100   # overlap to preserve context across chunk boundaries


# ─── Step 1: Load PDF ────────────────────────────────────────────────────────

def load_pdf(filepath: str = DEFAULT_PDF) -> str:
    """
    Extracts all text from a PDF file using PyMuPDF (fitz).

    Args:
        filepath: Path to the PDF file.

    Returns:
        Full extracted text as a single string.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ImportError:       If PyMuPDF is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"PDF not found at '{filepath}'. "
            "Place a research paper PDF in the papers/ directory."
        )

    doc = fitz.open(filepath)
    pages_text = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        # Skip near-empty pages (headers/footers only)
        if len(page_text.strip()) > 50:
            pages_text.append(page_text)

    doc.close()
    full_text = "\n".join(pages_text)

    # Light cleaning: collapse excessive whitespace
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    print(f"[RAG] Loaded PDF: {os.path.basename(filepath)}")
    print(f"[RAG] Total characters extracted: {len(full_text):,}")

    return full_text


# ─── Step 2: Create Chunks ───────────────────────────────────────────────────

def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Splits a long document into overlapping text chunks.

    WHY OVERLAP:
      Without overlap, sentences that cross a chunk boundary lose context.
      Overlap ensures every fact has a complete sentence before and after it,
      improving retrieval relevance.

    Args:
        text:       Full document text.
        chunk_size: Target size of each chunk in characters.
        overlap:    Number of characters to repeat between adjacent chunks.

    Returns:
        List of text chunk strings.
    """
    if not text or len(text) == 0:
        raise ValueError("Cannot create chunks from empty text.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # slide forward with overlap

    # Remove duplicate/near-empty chunks
    chunks = [c for c in chunks if len(c) > 30]

    print(f"[RAG] Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


# ─── Step 3: Create Embeddings ───────────────────────────────────────────────

def create_embeddings(
    chunks: List[str],
    model_name: str = EMBEDDING_MODEL,
) -> Tuple[np.ndarray, object]:
    """
    Encodes text chunks into dense vector embeddings.

    WHY SENTENCE TRANSFORMERS:
      Unlike word embeddings (Word2Vec), sentence transformers produce a single
      fixed-size vector for the entire sentence/chunk, capturing semantic meaning.
      'all-MiniLM-L6-v2' is small (80 MB), fast, and highly effective.

    Args:
        chunks:     List of text strings to embed.
        model_name: HuggingFace model identifier.

    Returns:
        (embeddings, model) where embeddings shape is (n_chunks, embedding_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    print(f"[RAG] Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    print(f"[RAG] Encoding {len(chunks)} chunks ...")
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
    )

    print(f"[RAG] Embedding shape: {embeddings.shape}")
    return np.array(embeddings, dtype=np.float32), model


# ─── Step 4: Build FAISS Index ───────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray):
    """
    Builds a FAISS flat index for exact nearest-neighbour search.

    WHY FAISS:
      FAISS (Facebook AI Similarity Search) is the industry-standard library
      for dense vector retrieval. For small datasets (< 10k chunks) we use
      IndexFlatIP (Inner Product = cosine similarity when vectors are normalised).
      For large datasets you'd use IndexIVFFlat or IndexHNSW for speed.

    Args:
        embeddings: Normalised embedding array (n_chunks, dim).

    Returns:
        faiss.IndexFlatIP ready for search.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Install faiss: pip install faiss-cpu")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine sim (vectors are normalised)
    index.add(embeddings)

    print(f"[RAG] FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index


# ─── Step 5: Search ──────────────────────────────────────────────────────────

def search(
    query: str,
    chunks: List[str],
    faiss_index,
    embed_model,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Encodes a query and retrieves the top-k most relevant chunks from the index.

    Args:
        query:       The user's question or search string.
        chunks:      The original text chunks (same order as when indexed).
        faiss_index: The FAISS index built by build_faiss_index().
        embed_model: The SentenceTransformer model used to encode chunks.
        top_k:       Number of results to return.

    Returns:
        List of (chunk_text, similarity_score) tuples, sorted by relevance.
    """
    import numpy as np

    # Embed and normalise the query using the same model and normalisation
    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True,
    ).astype(np.float32)

    # Search — D = distances (inner products), I = chunk indices
    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):  # guard against out-of-bounds
            results.append((chunks[idx], float(dist)))

    return results


# ─── High-Level RAG Pipeline ─────────────────────────────────────────────────

class RAGEngine:
    """
    Convenience wrapper that manages the full RAG lifecycle.

    Usage:
        engine = RAGEngine()
        engine.load("papers/sample.pdf")
        results = engine.query("What is the main finding?")
    """

    def __init__(self):
        self.chunks:      Optional[List[str]] = None
        self.index        = None
        self.embed_model  = None
        self._loaded_path: Optional[str] = None

    def load(self, filepath: str = DEFAULT_PDF) -> None:
        """
        Full indexing pipeline: PDF → chunks → embeddings → FAISS index.

        Idempotent: calling load() on the same file twice is a no-op.
        """
        if self._loaded_path == filepath:
            print("[RAG] Already loaded — skipping re-indexing.")
            return

        text = load_pdf(filepath)
        self.chunks = create_chunks(text)
        embeddings, self.embed_model = create_embeddings(self.chunks)
        self.index = build_faiss_index(embeddings)
        self._loaded_path = filepath

    def query(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieves the most relevant chunks for a given question.

        Args:
            question: Natural language query.
            top_k:    Number of results.

        Returns:
            List of (passage, score) tuples.
        """
        if self.index is None:
            raise RuntimeError("Call load() before query().")
        return search(question, self.chunks, self.index, self.embed_model, top_k)

    @property
    def is_loaded(self) -> bool:
        return self.index is not None
