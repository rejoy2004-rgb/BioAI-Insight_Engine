import os
import json
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = "data/faiss_index.bin"
DOCUMENTS_CACHE_PATH = "data/rag_documents.json"
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_index = None
_documents = []

def get_transformer_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def load_or_build_index(force=False):
    global _index, _documents
    
    if not force and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_CACHE_PATH):
        try:
            print("Loading cached RAG index and documents...")
            _index = faiss.read_index(FAISS_INDEX_PATH)
            with open(DOCUMENTS_CACHE_PATH, "r") as f:
                _documents = json.load(f)
            print(f"RAG index loaded successfully. Total documents: {len(_documents)}")
            return
        except Exception as e:
            print(f"Error loading cache: {e}. Rebuilding...")

    print("Rebuilding RAG index and documents...")
    documents = []
    
    # 1. Load synthetic papers corpus
    corpus_path = "data/papers_corpus.json"
    if os.path.exists(corpus_path):
        with open(corpus_path, "r") as f:
            corpus_papers = json.load(f)
        for paper in corpus_papers:
            text_to_encode = f"Title: {paper['title']}. Authors: {paper['authors']}. Journal: {paper['journal']} ({paper['year']}). Abstract: {paper['abstract']}"
            documents.append({
                "source": "Corpus Database",
                "title": paper["title"],
                "authors": paper["authors"],
                "journal": paper["journal"],
                "year": paper["year"],
                "abstract": paper["abstract"],
                "text_to_encode": text_to_encode
            })
            
    # 2. Parse sample.pdf chunks
    pdf_path = "papers/sample.pdf"
    if os.path.exists(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            pdf_text = ""
            for page in doc:
                pdf_text += page.get_text()
            
            # Split into chunks of 800 characters
            pdf_chunks = [pdf_text[i:i+800] for i in range(0, len(pdf_text), 800)]
            for idx, chunk in enumerate(pdf_chunks):
                text_to_encode = f"Title: Breast Cancer Reference (sample.pdf Chunk {idx+1}). Authors: Reference Study. Abstract: {chunk}"
                documents.append({
                    "source": "sample.pdf Reference Document",
                    "title": f"Breast Cancer Study Reference - Section {idx+1}",
                    "authors": "Kimberly Morton Cuthrell et al.",
                    "journal": "International Research Journal of Oncology",
                    "year": 2023,
                    "abstract": chunk.replace("\n", " ").strip(),
                    "text_to_encode": text_to_encode
                })
        except Exception as e:
            print(f"Error reading sample.pdf: {e}")
            
    if not documents:
        print("Warning: No documents found to index!")
        return

    print("Encoding documents using SentenceTransformers (this may take a few seconds)...")
    model = get_transformer_model()
    texts = [doc["text_to_encode"] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Create FAISS index
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save cache
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(DOCUMENTS_CACHE_PATH, "w") as f:
        json.dump(documents, f, indent=4)
        
    _index = index
    _documents = documents
    print(f"RAG index created successfully with {len(_documents)} documents and saved to disk.")

def retrieve(query, k=3):
    global _index, _documents
    if _index is None or not _documents:
        load_or_build_index()
        
    if _index is None or not _documents:
        return []
        
    model = get_transformer_model()
    query_embedding = model.encode([query]).astype("float32")
    
    D, I = _index.search(query_embedding, k)
    
    results = []
    for rank, idx in enumerate(I[0]):
        if idx == -1 or idx >= len(_documents):
            continue
        doc = _documents[idx]
        score = float(D[0][rank])
        
        # normalize score to similarity percentage
        similarity = 1 / (1 + score)
        
        results.append({
            "rank": rank + 1,
            "source": doc["source"],
            "title": doc["title"],
            "authors": doc["authors"],
            "journal": doc["journal"],
            "year": doc["year"],
            "abstract": doc["abstract"],
            "similarity_score": round(similarity, 4)
        })
        
    return results

if __name__ == "__main__":
    load_or_build_index(force=True)
    res = retrieve("What is the main finding regarding breast cancer screening and CNNs?", k=3)
    print("\nTest Retrieval Results:")
    for r in res:
        print(f"Rank {r['rank']}: {r['title']} (Score: {r['similarity_score']})")
        print(f"Abstract: {r['abstract'][:150]}...\n")