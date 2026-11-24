Optimize the RAG flow in this FastAPI service (ebook_ai_service) for speed on Render.
Constraints: do NOT use Supabase pgvector. Keep ChromaDB as the vector store, Supabase
Storage (bucket `chroma_db/` prefix) as where the index is stored, Gemini as the LLM,
and Render as the host. Flutter is the only client and calls this service over HTTPS.

Read CLAUDE.md first and respect its conventions: the pysqlite3 swap stays at the very
top of main.py, heavy imports stay lazy, the path-traversal guard in
load_database_from_supabase must be preserved, chroma_db_local/ and config/ stay gitignored.

## Current problems to fix
1. get_query_chain() calls load_database_from_supabase(LOCAL_DIR, force=True), so after
   every restart/cold start the first query re-downloads the whole index from Supabase.
2. Nothing is warmed up at startup: the first user request pays for downloading the
   index + loading the sentence-transformers model + building the chain.
3. Answers are returned only when fully generated (no streaming).
4. generate_and_store_database_task deletes chroma_db/ in Supabase and then uploads file
   by file, so a loader running at the same time can download a partial index.
5. /generate-database/ and /load-database/ are public; anyone can trigger them.

## Required changes

### A. Version-aware, download-once index loading (rag_service.py)
- Make the local index dir configurable: CHROMA_DIR env var, default ./chroma_db_local
  (so a Render persistent disk can be mounted there).
- Add ensure_local_index():
  - download only chroma_db/VERSION.json from Supabase first;
  - compare its "version" with the local VERSION.json; if equal, skip downloading;
  - if different or missing locally, download all files into a NEW temp dir next to
    CHROMA_DIR (same filesystem), verify chroma.sqlite3 + VERSION.json exist, then swap
    atomically (rename old -> .old, new -> CHROMA_DIR, delete .old);
  - if Supabase is unreachable but a local copy exists, log a warning and use the local copy.
- Remove force=True from the query path entirely. get_query_chain() must never download;
  it only opens whatever is in CHROMA_DIR.
- Protect index swap + chain rebuild with a threading.Lock so queries never see a
  half-swapped directory. Keep the current chain in a module-level variable that is
  replaced atomically (instead of relying only on lru_cache.cache_clear()).

### B. Safe publishing (generate_and_store_database_task)
- Upload all index files first, and upload VERSION.json LAST, so a loader only sees the
  new version when everything is in place. Delete stale remote files only after the
  new upload succeeds (files that are not part of the new index).
- After generating, update the local copy and swap the in-memory chain (no restart needed).

### C. Warm-up at startup (main.py)
- Use a FastAPI lifespan handler that starts a background thread (do not block port
  binding) which: runs ensure_local_index(), loads the embeddings model, builds the
  chain, and runs one dummy similarity search so everything is in memory.
- Track readiness state: "starting" | "ready" | "error" (+ error message).
- Add GET /health -> {"status": ..., "index_version": ..., "chunks": ...}.
- If a query arrives before ready, wait up to 30 s for readiness, then return 503
  {"detail": "Service warming up, retry shortly"}.

### D. Streaming query endpoint
- Add POST /query with JSON body {"query": str, "file_name": str | null}.
  - If file_name is given, filter retrieval with Chroma metadata filter
    {"file_name": file_name}.
  - Response is Server-Sent Events (text/event-stream):
      event: sources  data: {"sources": ["books/x.pdf", ...]}   (sent right after retrieval)
      event: token    data: {"text": "..."}                     (repeated, from llm.astream)
      event: done     data: {}
      event: error    data: {"detail": "..."}                   (on failure)
  - Run the retrieval (CPU-bound embedding + Chroma search) with run_in_threadpool;
    stream the Gemini answer with the async streaming API.
  - If no chunk passes the score threshold, stream a short "not found in the books"
    answer without calling the LLM.
- Keep GET /query-database/ working with the same response shape for backward
  compatibility, implemented on top of the same retrieval + generation code.
- Validate input: non-empty query, max 1000 chars.

### E. Reuse and timing
- The embeddings model, Chroma client, retriever and the Gemini client must each be
  created once per process and reused.
- Log per-stage timings for each query (retrieve_ms, first_token_ms, total_ms) at INFO.

### F. Protect admin endpoints
- /generate-database/ and /load-database/ require header X-Admin-Token equal to the
  ADMIN_TOKEN env var (constant-time compare with hmac.compare_digest). If ADMIN_TOKEN
  is not set, these endpoints return 503. /load-database/ calls ensure_local_index()
  (download only if the version changed) and swaps the chain.

### G. Render deployment
- In requirements.txt install the CPU-only torch wheel
  (--extra-index-url https://download.pytorch.org/whl/cpu) to cut memory/build size.
- Add a short "Deploying on Render" section to CLAUDE.md: env vars (GOOGLE_API_KEY,
  SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_BUCKET, ADMIN_TOKEN, CHROMA_DIR),
  optional persistent disk mounted at CHROMA_DIR, use /health as the health check path,
  and note that the free tier sleeps after ~15 min idle (keep-alive ping on /health or
  a paid instance avoids cold starts).
- Update the Architecture section of CLAUDE.md for the new endpoints and the flow.

### H. Flutter API contract
- Write docs/API.md describing POST /query (request body, SSE event format with an
  example), GET /health, and the 503 warming-up behaviour, so the Flutter app can
  consume the stream.

## Acceptance criteria
- Restarting the server with an unchanged remote index does NOT re-download any index
  files except VERSION.json (verify via logs).
- After the server reports ready on /health, a query makes zero Supabase Storage calls.
- POST /query streams a sources event and then tokens; the first token arrives before
  the full answer is generated.
- GET /query-database/ still returns {"response": "..."}.
- Running /generate-database/ while queries are in flight never causes a query error,
  and the new index is used afterwards without a restart.
- Admin endpoints reject requests without the correct X-Admin-Token.
- Add pytest tests (tests/) for: version comparison/skip-download logic (mock
  supabase_storage), the atomic swap, the admin token check, and the SSE event format
  (mock the retriever and the LLM). Add pytest to requirements-dev.txt.

Do not change ocr_service.py or the /upload endpoint. Do not commit .env, config/ or
chroma_db_local/. When done, summarize what changed and how to verify it locally.