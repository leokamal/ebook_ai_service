# ebook_ai_service

FastAPI service: OCR for uploaded images (EasyOCR) and RAG over PDF books (Chroma + Gemini). Storage is Cloudflare R2, with `data.json` in the bucket as the only database.

## Commands
- Install: `pip install -r requirements.txt` (`build.sh` does the same on deploy)
- Run: `python main.py` (honours `$PORT`, default 8000) or `uvicorn main:app --reload`
- No tests or linter configured yet.

## Environment variables (never hardcode)
- `GOOGLE_API_KEY` - Gemini
- `GEMINI_THINKING_LEVEL` - optional (`minimal` default, `low`, `medium`, `high`); higher is slower
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` - Cloudflare R2 token (Object Read & Write on one bucket, server-side only)
- `R2_BUCKET` - bucket name (default `ebook-ai`); PDFs under `books/`, covers under `images/`, the Chroma index under `chroma_db/<version>/`
- `ADMIN_TOKEN` - required header `X-Admin-Token` for `/generate-database/` and `/load-database/` (503 if unset)
- `CHROMA_DIR` - local index directory (default `./chroma_db_local`; mount a Render persistent disk here)
- `HF_TOKEN` - optional Hugging Face read token; `huggingface_hub` picks it up automatically (silences the "unauthenticated requests" warning, higher rate limits)
- `HF_HOME` - optional; point it at the persistent disk (e.g. `/data/hf`) so the embedding model isn't re-downloaded on each deploy

## Architecture
- `main.py` - FastAPI app. A lifespan handler starts a background warm-up thread. Endpoints: `GET /health`, `POST /query` (SSE stream), `GET /query-database/` (non-streaming), `POST /upload` (OCR), admin `POST /generate-database/` and `POST /load-database/`. See `docs/API.md`.
- `rag_service.py` - Startup uses the local index as-is (no R2 call); it only calls `ensure_local_index()` when no local index exists. `/load-database/` calls `ensure_local_index()` explicitly: it fetches only `data.json` and downloads the index only if the version changed (into a temp dir, sha256-checked, then swap). Queries use the local Chroma only (no R2 calls). `generate_and_store_database_task()` builds in a temp dir, uploads to `chroma_db/<version>/`, updates `data.json` LAST, swaps the local dir and in-memory state, then deletes all older index folders (only one index version is kept in R2). `_lock` guards the in-memory swap, `_update_lock` serializes download/generate.
- `security.py` - `verify_admin_token` dependency (constant-time compare).
- `ocr_service.py` - EasyOCR with preprocessing; readers cached per language.
- `r2_storage.py` - R2 wrapper (list, download, upload, delete) and `data.json` read/update (conditional write, backup first).
- `catalog.py` - keeps the `data.json` book list in step with the PDFs; hand-edited fields are preserved.
- `supabase_storage.py`, `scripts/migrate_supabase_to_r2.py` - one-off data migration only; delete after the switch is stable.
- Tests: `pip install -r requirements-dev.txt && pytest` (no network or models needed).

## Deploying on Render
- Build: `pip install -r requirements.txt` (CPU-only torch via the extra index). Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`. Set `PYTHON_VERSION=3.11.9`.
- Env vars: `GOOGLE_API_KEY`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`, `ADMIN_TOKEN`, `CHROMA_DIR`.
- Optional persistent disk mounted at `CHROMA_DIR` so restarts skip the download (no R2 call at startup).
- Health check path: `/health`.
- The free tier sleeps after ~15 min idle (cold start = reload model + index). Ping `/health` periodically or use a paid instance. Torch + sentence-transformers need roughly 1-2 GB RAM.

## Conventions / gotchas
- When a choice has options, take the faster one for the API (mind the host's memory limit).
- Heavy imports (torch, easyocr, chroma, gemini) are lazy, inside functions, so the server binds its port fast. Keep it that way.
- `main.py` swaps `pysqlite3` in as `sqlite3` before chromadb is imported; keep that block at the very top.
- `chroma_db_local/` and `config/` are gitignored and must not be committed.
- Downloaded storage object names are path-checked before writing to disk; preserve that traversal guard.
- CORS is currently `*`; tighten before production.
