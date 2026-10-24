# Retrieval evaluation

## What the system is (and is not)

* **Corpus:** one real, open-access paper - Cuthrell & Tzenios, *Breast Cancer: Updated and Deep Insights*, Int. Res. J. Oncol. 6(1), 104-118 (2023), Article no. IRJO.100349 (`papers/sample.pdf`; the PDF itself states a CC BY 4.0 licence). 15 pages -> **61 passages** of ~130 words (30-word overlap) after removing running headers/footers and the reference list.
* **Removed:** the 510-document synthetic corpus of the original project (see INITIAL_AUDIT #1). It described fake studies with real journal names.
* **Retrieval:** default = word 1-2-gram TF-IDF with cosine similarity (scikit-learn). This is *lexical* retrieval. Optional = `BIOAI_RETRIEVER=dense` -> `all-MiniLM-L6-v2` embeddings with cosine similarity (numpy; FAISS is unnecessary for 61 vectors). Embeddings are cached by hash of model name + passages.
* **Answers:** *extractive*. Up to three sentences that overlap the question are quoted verbatim with `[n]` citations pointing at source cards. There is **no LLM and no generated text**, so titles/authors/dates cannot be invented; every card comes from `papers/sources.json`. Abstains (`no_evidence`) when no passage clears the minimum score or no sentence matches.
* **Not implemented:** reranking, query rewriting, multi-document conflict detection, LLM generation. With one source, "conflicting evidence" cannot arise; the answer disclaimer tells the reader that quoted statements may be associations or opinions.

## Measurements

`python scripts/eval_rag.py --dense` -> raw output in [`rag_eval_output.json`](rag_eval_output.json). 20 hand-written questions over the single document; a retrieved passage is "relevant" if it matches a regex of terms that an answering passage must contain (e.g. `BRCA`, `metasta`). 12 questions reuse the passage's vocabulary ("lexical"), 8 are paraphrases with little word overlap.

| Method | hit@1 | hit@3 | MRR@3 | hit@3 lexical | hit@3 paraphrase | median latency |
|---|---|---|---|---|---|---|
| TF-IDF (default) | 0.60 | 0.65 | 0.625 | 0.83 | 0.375 | 0.6 ms |
| Dense MiniLM | **0.80** | **0.90** | **0.850** | 1.00 | 0.75 | 12.9 ms |
| Hybrid (RRF of both) | 0.60 | 0.80 | 0.692 | 0.92 | 0.625 | 11.6 ms |

**How to read this.** The set is tiny (20 queries, 1 document) and the relevance labels are keyword-based, which if anything favours TF-IDF - yet dense retrieval still wins by 25 points hit@3, mostly on paraphrases. So: dense is **measurably better**; TF-IDF is the default only because it needs no model download, no torch (hundreds of MB in the Docker image) and answers in <1 ms. If the demo is run with internet access and `sentence-transformers` installed, use `BIOAI_RETRIEVER=dense`. The simple RRF hybrid did not help here. These numbers do not generalise to other corpora.

### Abstention threshold

* TF-IDF: cosine >= 0.08, and the query must share at least one vocabulary term with the corpus.
* Dense: cosine >= 0.25. On this corpus 20 on-topic queries had a best-passage cosine of at least 0.33 (median 0.60) while 10 unrelated queries (baking, football, finance...) scored <= 0.17. **Not tested:** biomedical questions the paper does not cover (e.g. "diabetes insulin resistance" scored 0.28 and returned a passage; the extractive step then found no matching sentence and answered `no_evidence`).

## Behaviour verified by automated tests (`tests/test_rag.py`, `tests/test_api.py`)

empty / whitespace / over-long queries (422); empty corpus; missing manifest; malformed manifest JSON; entries without id/title; missing files; documents with no text; duplicate documents (deduplicated); path traversal in the manifest (blocked); header/reference stripping; chunk overlap/coverage; ranking sanity; repeated query determinism; k clamping; latency smoke over 200 documents; every cited quote is a verbatim substring of the cited passage; prompt-injection text inside a document is only ever returned as quoted source text; retrieval exceptions return a generic 500 with a request id; dense backend falls back visibly when unavailable.

## Known weaknesses

* Quoted sentences can start mid-heading (e.g. "1.2.2 Factors related to genetics Complex acquired..."), because PDF text extraction loses layout.
* Sentence selection is word-overlap based, so it can pick a topically related but non-answering sentence.
* One paper is a thin evidence base; add documents via `papers/sources.json` (PDF/TXT/MD, metadata required) and re-run `scripts/eval_rag.py` with new gold queries.
