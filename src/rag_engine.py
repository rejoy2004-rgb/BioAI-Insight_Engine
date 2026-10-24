"""Lightweight scientific retrieval (default: scikit-learn only; optional dense embeddings).

Default: chunk-level TF-IDF (word 1-2 grams, sublinear tf, cosine
similarity) over the documents listed in papers/sources.json. It is *lexical*
retrieval, not semantic search. Set BIOAI_RETRIEVER=dense (needs sentence-transformers) for MiniLM
embeddings, which scored higher in the small evaluation in docs/RAG_EVALUATION.md.

Answers are *extractive*: sentences are quoted verbatim from retrieved chunks
and cited by source number. No text is generated, so nothing can be invented;
if no chunk matches well enough the engine abstains. Retrieved text is treated
as untrusted data and never interpreted as instructions.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config

log = logging.getLogger("bioai.rag")

CHUNK_WORDS = 130
CHUNK_OVERLAP = 30
MIN_SCORE = 0.08          # cosine below this => treated as no evidence
MAX_TEXT_CHARS = 2_000_000
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "and", "or", "to", "is", "are", "what", "how", "for", "with", "does", "do"}


class QueryError(ValueError):
    """Invalid user query (empty / too long)."""


def clean_text(text: str) -> str:
    return _WS.sub(" ", _CTRL.sub(" ", text or "")).strip()


def normalize_query(q: str) -> str:
    return clean_text(q)


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if size <= overlap:
        raise ValueError("size must exceed overlap")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page: int | None
    meta: dict


@dataclass
class LoadReport:
    documents_loaded: int = 0
    chunks: int = 0
    duplicates_skipped: int = 0
    warnings: list[str] = field(default_factory=list)


_REF_HEADING = re.compile(r"^\s*(references|bibliography)\s*$", re.I)


def strip_boilerplate(pages: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    """Drop running headers/footers (lines recurring on >=40% of pages, digits
    normalised) and everything from a 'REFERENCES' heading onwards, so that
    bibliography entries are not returned as evidence."""
    if not pages:
        return pages
    norm = lambda ln: re.sub(r"\d+", "#", ln.strip().lower())
    freq: dict[str, int] = {}
    for _, text in pages:
        for ln in {norm(ln) for ln in text.splitlines() if ln.strip()}:
            freq[ln] = freq.get(ln, 0) + 1
    limit = max(2, int(0.4 * len(pages)))
    out, in_refs = [], False
    for no, text in pages:
        kept = []
        for ln in text.splitlines():
            if _REF_HEADING.match(ln):
                in_refs = True
            if in_refs:
                break
            if len(pages) >= 3 and freq.get(norm(ln), 0) >= limit:
                continue
            kept.append(ln)
        out.append((no, " ".join(kept)))
        if in_refs:
            break
    return out


def _pdf_pages(path: Path) -> list[tuple[int, str]]:
    import fitz  # PyMuPDF
    with fitz.open(path) as doc:
        return strip_boilerplate([(i + 1, page.get_text()) for i, page in enumerate(doc)])


def load_chunks(sources_path: Path | str = config.SOURCES_PATH,
                base_dir: Path | str | None = None) -> tuple[list[Chunk], LoadReport]:
    """Read the manifest and build chunks. Problems become warnings, never silent."""
    sources_path = Path(sources_path)
    base = Path(base_dir) if base_dir else sources_path.parent
    report = LoadReport()
    if not sources_path.exists():
        report.warnings.append("sources manifest not found")
        return [], report
    try:
        manifest = json.loads(sources_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, list):
            raise ValueError("manifest must be a JSON list")
    except (ValueError, OSError) as exc:
        report.warnings.append(f"invalid sources manifest: {type(exc).__name__}")
        return [], report

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for i, src in enumerate(manifest):
        if not isinstance(src, dict) or not src.get("id") or not src.get("title"):
            report.warnings.append(f"source #{i}: missing 'id' or 'title'; skipped")
            continue
        meta = {k: src.get(k) for k in ("title", "authors", "journal", "year", "identifier", "url", "license")}
        meta["synthetic"] = bool(src.get("synthetic", False))
        meta["doc_id"] = src["id"]
        pages: list[tuple[int | None, str]] = []
        try:
            if "text" in src:                      # inline text record
                pages = [(None, str(src["text"]))]
            else:
                path = (base / str(src.get("file", ""))).resolve()
                if base.resolve() not in path.parents:
                    raise ValueError("path escapes papers directory")
                if path.suffix.lower() == ".pdf":
                    pages = _pdf_pages(path)
                elif path.suffix.lower() in {".txt", ".md"}:
                    pages = [(None, path.read_text(encoding="utf-8")[:MAX_TEXT_CHARS])]
                else:
                    raise ValueError("unsupported file type")
        except Exception as exc:
            report.warnings.append(f"{src['id']}: could not read ({type(exc).__name__}); skipped")
            continue
        n_before = len(chunks)
        for page_no, raw in pages:
            for j, text in enumerate(chunk_words(clean_text(raw))):
                digest = hashlib.sha1(text.lower().encode()).hexdigest()
                if digest in seen:
                    report.duplicates_skipped += 1
                    continue
                seen.add(digest)
                chunks.append(Chunk(f"{src['id']}:p{page_no}:{j}", src["id"], text, page_no, meta))
        if len(chunks) == n_before:
            report.warnings.append(f"{src['id']}: no extractable text; skipped")
        else:
            report.documents_loaded += 1
    report.chunks = len(chunks)
    return chunks, report


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._vec = None
        self._matrix = None
        if chunks:
            self._vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                                        sublinear_tf=True, min_df=1)
            self._matrix = self._vec.fit_transform([c.text for c in chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        query = normalize_query(query)
        if not query:
            raise QueryError("Query is empty.")
        if len(query) > config.MAX_QUERY_CHARS:
            raise QueryError(f"Query exceeds {config.MAX_QUERY_CHARS} characters.")
        k = max(1, min(int(k), 10))
        if self._matrix is None:
            return []
        q = self._vec.transform([query])
        if q.nnz == 0:           # no query term occurs in the corpus vocabulary
            return []
        scores = (self._matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:k]
        results = []
        for idx in order:
            if scores[idx] < MIN_SCORE:
                break
            c = self.chunks[idx]
            results.append({"rank": len(results) + 1, "chunk_id": c.chunk_id, "doc_id": c.doc_id,
                            "page": c.page, "score": round(float(scores[idx]), 4), "text": c.text, **c.meta})
        return results


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.split()) >= 6]


def build_answer(query: str, results: list[dict], max_sentences: int = 3) -> dict:
    """Extractive, cited answer. Every sentence is a verbatim substring of a retrieved chunk."""
    if not results:
        return {"status": "no_evidence", "answer": None,
                "message": "No sufficiently relevant passage was found in the indexed documents. "
                           "This does not mean the answer is unknown, only that it is not covered by this corpus.",
                "citations": [], "sources": []}
    q_terms = set(_TOKEN.findall(query.lower())) - _STOP
    scored = []
    for r in results:
        for s in _sentences(r["text"]):
            overlap = len(q_terms & set(_TOKEN.findall(s.lower())))
            if overlap:
                scored.append((overlap, r["score"], s, r))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    picked, used = [], set()
    for _overlap, _, sent, r in scored:
        if sent in used:
            continue
        used.add(sent)
        picked.append((sent, r))
        if len(picked) >= max_sentences:
            break
    if not picked:
        return {"status": "no_evidence", "answer": None,
                "message": "Retrieved passages did not contain a sentence directly matching the question.",
                "citations": [], "sources": []}
    source_ids = {}
    sources = []
    for _, r in picked:
        if r["chunk_id"] not in source_ids:
            source_ids[r["chunk_id"]] = len(sources) + 1
            sources.append({"n": len(sources) + 1, **{k: r[k] for k in (
                "chunk_id", "doc_id", "title", "authors", "journal", "year", "identifier", "url",
                "page", "score", "synthetic")}})
    citations = [{"n": source_ids[r["chunk_id"]], "quote": sent} for sent, r in picked]
    return {
        "status": "ok",
        "answer": " ".join(f"{c['quote']} [{c['n']}]" for c in citations),
        "message": "Extractive answer: sentences are quoted verbatim from the cited passages. Quoted statements "
                   "may describe associations, reviews or opinions; check the source for study design before "
                   "drawing causal conclusions. Not reviewed by a clinician or scientist.",
        "citations": citations,
        "sources": sources,
    }


DENSE_MODEL = "all-MiniLM-L6-v2"
DENSE_MIN_SCORE = 0.25    # cosine; on-topic queries scored >=0.33 and off-topic <=0.17 in docs/RAG_EVALUATION.md


class DenseRetriever:
    """Optional semantic retrieval with sentence-transformers (numpy cosine; no FAISS needed at this scale).

    Embeddings are cached in data/rag_cache/ keyed by model name + a hash of the chunk texts, so indexing is
    reproducible and only recomputed when the corpus changes. Requires `pip install sentence-transformers`.
    """

    def __init__(self, chunks: list[Chunk], model_name: str = DENSE_MODEL, cache_dir: Path | None = None):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.chunks, self.model_name = chunks, model_name
        self._model = SentenceTransformer(model_name)
        digest = hashlib.sha256((model_name + "|".join(c.text for c in chunks)).encode()).hexdigest()[:16]
        cache_dir = Path(cache_dir or config.DATA_DIR / "rag_cache")
        cache = cache_dir / f"emb-{digest}.npy"
        if cache.exists():
            self._emb = np.load(cache)
        else:
            self._emb = self._model.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False)
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(cache, self._emb)
            except OSError:
                log.warning("could not write embedding cache")

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        query = normalize_query(query)
        if not query:
            raise QueryError("Query is empty.")
        if len(query) > config.MAX_QUERY_CHARS:
            raise QueryError(f"Query exceeds {config.MAX_QUERY_CHARS} characters.")
        k = max(1, min(int(k), 10))
        if not len(self.chunks):
            return []
        scores = self._emb @ self._model.encode([query], normalize_embeddings=True)[0]
        results = []
        for idx in np.argsort(-scores, kind="stable")[:k]:
            if scores[idx] < DENSE_MIN_SCORE:
                break
            c = self.chunks[idx]
            results.append({"rank": len(results) + 1, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "page": c.page,
                            "score": round(float(scores[idx]), 4), "text": c.text, **c.meta})
        return results


class RagEngine:
    """Holds the index; build once at startup."""

    def __init__(self, sources_path: Path | str = config.SOURCES_PATH):
        self.sources_path = Path(sources_path)
        self.retriever = Retriever([])
        self.report = LoadReport()
        self.error: str | None = None
        self.method = "TF-IDF (lexical), chunk-level"

    @property
    def ready(self) -> bool:
        return len(self.retriever) > 0

    def build(self) -> None:
        t0 = time.perf_counter()
        try:
            chunks, self.report = load_chunks(self.sources_path)
            self.retriever = self._make_retriever(chunks)
            self.error = None if chunks else "No documents indexed."
        except Exception as exc:
            self.error = f"Index build failed ({type(exc).__name__})."
            log.error(self.error)
        log.info("rag index: %d chunks, %d docs, %.0f ms, warnings=%s", len(self.retriever),
                 self.report.documents_loaded, (time.perf_counter() - t0) * 1000, self.report.warnings)

    def _make_retriever(self, chunks: list[Chunk]):
        wanted = os.getenv("BIOAI_RETRIEVER", "tfidf").lower()
        self.method = "TF-IDF (lexical), chunk-level"
        if wanted == "dense" and chunks:
            try:
                r = DenseRetriever(chunks)
                self.method = f"dense embeddings ({DENSE_MODEL}), chunk-level"
                return r
            except Exception as exc:   # missing package / model download failure -> fall back, visibly
                msg = f"dense retriever unavailable ({type(exc).__name__}); using TF-IDF"
                self.report.warnings.append(msg)
                log.warning(msg)
        return Retriever(chunks)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        return self.retriever.search(query, k)

    def ask(self, query: str, k: int = 3) -> dict:
        results = self.retrieve(query, k)
        return {**build_answer(query, results), "retrieved": results}

    def stats(self) -> dict:
        titles = sorted({c.meta["title"] for c in self.retriever.chunks})
        return {"chunks": len(self.retriever), "documents": self.report.documents_loaded, "titles": titles,
                "warnings": self.report.warnings, "retrieval_method": self.method,
                "answer_method": "extractive, cited"}
