# Security and privacy notes

Scope: a single-process demo service with **no authentication, no database, no uploads and no external API calls**. The review below is sized to that attack surface.

## Data flow and external services

| Data | Goes where | Notes |
|---|---|---|
| Nine biopsy scores (`/predict`, `/predict/batch`) | in-process only | Not logged, not stored. Request logs contain method, path, status, timing and a random request id. |
| Research questions (`/rag/*`) | in-process only | Not logged (only result counts / status). |
| Browser -> Google Fonts | stylesheet request only | The only third-party request; the page works without it (fallback fonts). No user input is sent. |
| Hugging Face | only if `BIOAI_RETRIEVER=dense`: one-time model download | No user data sent. |
| LLM providers | **none** | The project calls no LLM, so no health-related text can reach one. If a generator is added later, send only the retrieved passages + question, never classifier inputs. |

No API keys are used; `.env.example` contains no secrets. A pattern scan of the tree and the single git commit found no credentials (the upstream GitHub history was not available to me - **if you ever committed a key there, rotate it**: deleting the file does not revoke it).

## Controls implemented (each has a test unless noted)

| Risk | Control |
|---|---|
| Malformed / hostile JSON | Pydantic models, `extra="forbid"`, whole numbers 1-10, NaN/Infinity rejected, 422 with field + message only (input is **not** echoed back) |
| Oversized requests | 512 KiB body cap (413), batch <= 1000 rows, query <= 500 chars, `k` <= 10 |
| Stack traces / paths in errors | Global handler returns `{"detail": "Internal server error.", "request_id"}`; trace goes to logs only |
| XSS | Frontend never builds HTML from data (`textContent`/`createElement` only; a test bans `innerHTML` etc.); a browser test submits an `<img onerror>` payload and verifies it does not run; strict CSP `script-src 'self'`, no inline scripts |
| Third-party script risk | Chart.js and Font Awesome are vendored under `static/vendor/` (pinned versions); no CDN scripts |
| Unsafe deserialisation | The model bundle is loaded with `joblib` **only after its SHA-256 matches `metadata.json`**; the app never loads user-supplied pickles; there are no upload endpoints for models. Caveat: the hash lives next to the file, so it detects accidental corruption/partial writes, not an attacker who can write both files. |
| Path traversal | Static files via Starlette `StaticFiles` (tested with `../` and `%2e%2e`); the RAG manifest cannot reference files outside `papers/` (tested) |
| State-changing endpoint | `/train` is **absent unless `BIOAI_ENABLE_TRAIN_ENDPOINT=1`**; single-flight lock when enabled. It is unauthenticated - never enable it on a shared host |
| CORS | No CORS middleware: same-origin only (tested that no `Access-Control-Allow-Origin` is emitted) |
| Clickjacking / sniffing | `frame-ancestors 'none'`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` |
| Debug mode | uvicorn is started without `--reload`; no debug flag |
| Container | Runs as non-root user (Dockerfile; **not built here**) |
| Prompt injection | There is no instruction-following component. Retrieved text is only quoted; a test places "ignore all previous instructions..." in a document and checks it is only returned as source text |
| Memory growth | Index is built once; no per-request caches; request bodies capped |

## Not implemented / known gaps

* **No authentication or rate limiting** - fine for a local demo, not for the public internet. Put it behind a reverse proxy with auth and rate limits if exposed.
* `/docs` (Swagger UI) loads assets from a CDN and therefore gets a relaxed CSP.
* No automated dependency-vulnerability scan was run (`pip-audit` was not installed/executed); versions are bounded, not pinned.
* HTTPS/HSTS must be provided by the deployment environment.
* Chunk text from PDFs is treated as untrusted but is rendered only through `textContent`; PDF parsing itself (PyMuPDF) is applied only to the repository's own PDF - **do not point `sources.json` at untrusted PDFs** without sandboxing.
