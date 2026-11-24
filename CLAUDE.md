# ebook_ai_service

FastAPI service: OCR for uploaded images (EasyOCR) and RAG over PDF books (Chroma + Gemini). Storage is Supabase Storage.

## Commands
- Install: `pip install -r requirements.txt` (`build.sh` does the same on deploy)
- Run: `python main.py` (honours `$PORT`, default 8000) or `uvicorn main:app --reload`
- No tests or linter configured yet.

## Environment variables (never hardcode)
- `GOOGLE_API_KEY` – Gemini
- `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` – Supabase project (service_role key, server-side only)
- `SUPABASE_BUCKET` – bucket name (default `ebook-ai`); PDFs live under `books/`, the Chroma index under `chroma_db/`

## Architecture
- `main.py` – endpoints: `/upload` (OCR), `/query-database/`, `/generate-database/` (background), `/load-database/`.
- `rag_service.py` – builds the Chroma index from `books/*.pdf`, uploads it to `chroma_db/`, downloads it to `./chroma_db_local`, and builds the cached query chain.
- `ocr_service.py` – EasyOCR with preprocessing; readers cached per language.
- `supabase_storage.py` – Supabase Storage wrapper (recursive list, download, upload, delete).

## Conventions / gotchas
- Heavy imports (torch, easyocr, chroma, gemini) are lazy, inside functions, so the server binds its port fast. Keep it that way.
- `main.py` swaps `pysqlite3` in as `sqlite3` before chromadb is imported; keep that block at the very top.
- `chroma_db_local/` and `config/` are gitignored and must not be committed.
- Downloaded storage object names are path-checked before writing to disk; preserve that traversal guard.
- CORS is currently `*`; tighten before production.
