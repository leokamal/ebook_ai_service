# ebook_ai_service

FastAPI service: OCR for uploaded images (EasyOCR) and RAG over PDF books (Chroma + Gemini). Storage is Supabase Storage.

## Commands
- Install: `pip install -r requirements.txt` (`build.sh` does the same on deploy)
- Run: `python main.py` (honours `$PORT`, default 8000) or `uvicorn main:app --reload`
- No tests or linter configured yet.

## Environment variables (never hardcode)
- `GOOGLE_API_KEY` - Gemini
- `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` - Supabase project (secret/service_role key, server-side only)
- `SUPABASE_BUCKET` - bucket name (default `ebook-ai`); PDFs under `books/`, the Chroma index under `chroma_db/`
- `ADMIN_TOKEN` - required header `X-Admin-Token` for `/generate-database/` and `/load-database/` (503 if unset)
- `CHROMA_DIR` - local index directory (default `./chroma_db_local`; mount a Render persistent disk here)

## Architecture
- `main.py` - FastAPI app. A lifespan handler starts a background warm-up thread. Endpoints: `GET /health`, `POST /query` (SSE stream), `GET /query-database/` (non-streaming), `POST /upload` (OCR), admin `POST /generate-database/` and `POST /load-database/`. See `docs/API.md`.
- `rag_service.py` - `ensure_local_index()` fetches only `chroma_db/VERSION.json` and downloads the index only if the version changed (into a temp dir, then atomic swap). Queries use the local Chroma only (no Supabase calls). `generate_and_store_database_task()` builds in a temp dir, uploads `VERSION.json` LAST, deletes stale remote files, then swaps the local dir and in-memory state. `_lock` guards the in-memory swap, `_update_lock` serializes download/generate.
- `security.py` - `verify_admin_token` dependency (constant-time compare).
- `ocr_service.py` - EasyOCR with preprocessing; readers cached per language.
- `supabase_storage.py` - Supabase Storage wrapper (recursive list, download, upload, delete).
- Tests: `pip install -r requirements-dev.txt && pytest` (no network or models needed).

## Deploying on Render
- Build: `pip install -r requirements.txt` (CPU-only torch via the extra index). Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`. Set `PYTHON_VERSION=3.11.9`.
- Env vars: `GOOGLE_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SUPABASE_BUCKET`, `ADMIN_TOKEN`, `CHROMA_DIR`.
- Optional persistent disk mounted at `CHROMA_DIR` so restarts skip the download (only `VERSION.json` is fetched).
- Health check path: `/health`.
- The free tier sleeps after ~15 min idle (cold start = reload model + index). Ping `/health` periodically or use a paid instance. Torch + sentence-transformers need roughly 1-2 GB RAM.

## Conventions / gotchas
- Heavy imports (torch, easyocr, chroma, gemini) are lazy, inside functions, so the server binds its port fast. Keep it that way.
- `main.py` swaps `pysqlite3` in as `sqlite3` before chromadb is imported; keep that block at the very top.
- `chroma_db_local/` and `config/` are gitignored and must not be committed.
- Downloaded storage object names are path-checked before writing to disk; preserve that traversal guard.
- CORS is currently `*`; tighten before production.
