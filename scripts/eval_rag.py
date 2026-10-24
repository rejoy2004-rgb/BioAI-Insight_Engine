"""Small, reproducible retrieval evaluation: TF-IDF (shipped) vs dense MiniLM (optional baseline).

    python scripts/eval_rag.py            # TF-IDF only
    python scripts/eval_rag.py --dense    # also needs `pip install sentence-transformers` (downloads all-MiniLM-L6-v2)

Relevance judgement: a retrieved chunk counts as relevant if it matches the query's
`gold` regex (terms that must appear in a passage that answers the question).
This is a weak, keyword-based label that favours lexical retrieval, and the set is tiny
(one document, 20 queries) - see docs/RAG_EVALUATION.md for how to read the numbers.
`kind`: "lexical" queries share words with the answer passage; "paraphrase" queries
deliberately avoid them.
"""
try:  # torch first: avoids a Windows DLL-order failure when scikit-learn is imported before torch
    import torch  # noqa: F401
except ImportError:
    pass
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.rag_engine import Retriever, load_chunks  # noqa: E402

QUERIES = [
    # lexical
    ("lexical", "What role do BRCA1 and BRCA2 play in breast cancer?", r"BRCA"),
    ("lexical", "triple negative breast cancer", r"triple"),
    ("lexical", "How does mammography screening work?", r"mammograph"),
    ("lexical", "lymph node involvement in breast cancer", r"lymph"),
    ("lexical", "breast cancer metastasis", r"metasta"),
    ("lexical", "What is ductal carcinoma in situ?", r"ductal carcinoma in situ|DCIS"),
    ("lexical", "hormone receptor and breast cancer", r"hormone"),
    ("lexical", "radiation therapy for breast cancer", r"radiation"),
    ("lexical", "chemotherapy for breast cancer", r"chemotherap"),
    ("lexical", "obesity and breast cancer risk", r"obes"),
    ("lexical", "global incidence of breast cancer", r"incidence"),
    ("lexical", "HER2 positive tumours", r"HER2"),
    # paraphrase (little word overlap with the target passage)
    ("paraphrase", "inherited gene faults that raise the chance of disease", r"BRCA|genetic"),
    ("paraphrase", "using X-ray imaging of the breast to find tumours early", r"mammograph|screening"),
    ("paraphrase", "tumour spreading to distant organs", r"metasta"),
    ("paraphrase", "drug treatment given before or after an operation", r"chemotherap|adjuvant|neoadjuvant"),
    ("paraphrase", "excess body weight as a danger factor", r"obes|weight|BMI"),
    ("paraphrase", "number of new cases diagnosed worldwide each year", r"incidence|million new"),
    ("paraphrase", "how far the disease has progressed at diagnosis", r"stage|staging"),
    ("paraphrase", "women whose periods began early or ended late", r"menopaus|menstruat"),
]


def evaluate(search, name, k=3):
    hits1, hits3, rr, lat = [], [], [], []
    per_kind = {}
    for kind, q, gold in QUERIES:
        t = time.perf_counter()
        results = search(q, k)
        lat.append((time.perf_counter() - t) * 1000)
        rel = [bool(re.search(gold, r, re.I)) for r in results]
        h1, h3 = bool(rel[:1] and rel[0]), any(rel[:3])
        r_ = next((1 / (i + 1) for i, x in enumerate(rel) if x), 0.0)
        hits1.append(h1)
        hits3.append(h3)
        rr.append(r_)
        d = per_kind.setdefault(kind, [[], []])
        d[0].append(h3)
        d[1].append(r_)
    out = {"method": name, "hit@1": float(np.mean(hits1)), "hit@3": float(np.mean(hits3)), "mrr@3": float(np.mean(rr)),
           "median_latency_ms": round(statistics.median(lat), 2),
           "by_kind": {k: {"hit@3": float(np.mean(v[0])), "mrr@3": float(np.mean(v[1]))} for k, v in per_kind.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true")
    args = ap.parse_args()
    chunks, _ = load_chunks()
    tfidf = Retriever(chunks)
    results = [evaluate(lambda q, k: [r["text"] for r in tfidf.search(q, k)], "tfidf")]
    if args.dense:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode([c.text for c in chunks], normalize_embeddings=True)

        def dense(q, k):
            s = emb @ model.encode([q], normalize_embeddings=True)[0]
            return [chunks[i].text for i in np.argsort(-s)[:k]]
        results.append(evaluate(dense, "dense all-MiniLM-L6-v2 (no score threshold)"))
        # hybrid: reciprocal-rank fusion of the two rankings
        def hybrid(q, k):
            s = emb @ model.encode([q], normalize_embeddings=True)[0]
            d_rank = {int(i): r for r, i in enumerate(np.argsort(-s)[:20])}
            t_scores = (tfidf._matrix @ tfidf._vec.transform([q]).T).toarray().ravel()
            t_rank = {int(i): r for r, i in enumerate(np.argsort(-t_scores)[:20])}
            fused = {i: 1 / (60 + d_rank.get(i, 99)) + 1 / (60 + t_rank.get(i, 99)) for i in set(d_rank) | set(t_rank)}
            return [chunks[i].text for i in sorted(fused, key=fused.get, reverse=True)[:k]]
        results.append(evaluate(hybrid, "hybrid (RRF of tfidf + dense)"))
    print(json.dumps({"n_queries": len(QUERIES), "n_chunks": len(chunks), "results": results}, indent=2))


if __name__ == "__main__":
    main()
