# Plan: migrate from Supabase Storage to Cloudflare R2

Status: code implemented (phases 2 and 4, code only). Buckets, data migration and deployment are not done. Replaces `migrate_supabase_to_github.md`.

Differences from this plan, found while implementing:

- Supabase also has a `public.books` table with public read access (`supabase/migrations/`). The migration script copies its rows into `data.json`, adding `lang` and `legacy_id`.
- Covers keep their existing `images/...` keys, not `images/covers/`.
- `supabase_storage.py`, `.mcp.json` and `supabase/` stay until phase 6.

## 1. Goal

- One private R2 bucket holds all files: PDFs, images and the search index.
- One `data.json` object in that bucket is the only database.
- Supabase is removed from the service.

## 2. Where we are today

- The service uses Supabase **Storage only**. The code makes no Postgres table calls.
- `books/*.pdf` are the source books. `chroma_db/` is the published index, versioned by `chroma_db/VERSION.json`.
- `/upload` runs OCR in memory and stores nothing.
- All storage access goes through four functions in `supabase_storage.py`: `list_files`, `download_bytes`, `upload_file`, `delete_files`.
- Queries use the local Chroma copy only. This stays true.

## 3. Target design

### 3.1 Bucket layout

Bucket name: `ebook-ai`.

```
ebook-ai/
├── data.json
├── books/<name>.pdf            # same keys as today
├── images/covers/<id>.jpg
├── chroma_db/<version>/...     # one folder per published index
└── backups/data-<version>.json
```

- Keep the `books/<name>.pdf` keys unchanged: they are stored as `file_name` in chunk metadata and the Flutter app sends them back to `/query`.
- Each index gets its own folder, so publishing never overwrites files that another server is downloading.

### 3.2 `data.json`

```json
{
  "schema_version": 1,
  "updated_at": "2026-11-25T10:00:00Z",
  "books": [
    {
      "id": "flutter-succinctly",
      "title": "Flutter Succinctly",
      "file_name": "books/flutter-succinctly.pdf",
      "cover": "images/covers/flutter-succinctly.jpg",
      "size_bytes": 1234567,
      "etag": "<object etag>",
      "indexed": true,
      "added_at": "2026-11-25T10:00:00Z"
    }
  ],
  "index": {
    "version": "20261125T100000Z",
    "prefix": "chroma_db/20261125T100000Z",
    "files": [{ "path": "chroma.sqlite3", "sha256": "<hex>" }],
    "chunks": 812,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

Rules:

- `data.json.index` replaces the remote `VERSION.json`. A local `VERSION.json` is still written in `CHROMA_DIR` with today's fields, so `/health` does not change.
- The service writes `index`, `size_bytes`, `etag` and `indexed`. It must preserve hand-edited fields (`title`, `cover`).
- A PDF in `books/` with no entry gets one automatically. An entry whose PDF is gone is removed.
- Writes are conditional: send `If-Match` with the ETag that was read (`If-None-Match: *` for the first write). On HTTP 412, re-read and retry, at most 3 times.
- Before replacing `data.json`, copy the old one to `backups/`. Keep the last 10.
- No secrets or personal user data.

One JSON file suits a small catalog with few writes. If user accounts, favourites or reading progress are needed later, add a real database (for example Cloudflare D1).

### 3.3 R2 facts that shape the design

| Item | Value |
|---|---|
| Free tier per month (Standard storage) | 10 GB-month, 1 million Class A (writes, lists), 10 million Class B (reads) |
| Above the free tier | $0.015 per GB-month, $4.50 per million Class A, $0.36 per million Class B |
| Download (egress) fees | None |
| Conditional `PutObject` | `If-Match` and `If-None-Match` supported |
| API | S3-compatible |

Sources: [R2 pricing](https://developers.cloudflare.com/r2/pricing), [conditional operations](https://developers.cloudflare.com/r2/api/s3/conditional-operations/), [S3 compatibility](https://developers.cloudflare.com/r2/api/s3/api/). Taken from search results on 2026-11-25; confirm on those pages.

Not yet confirmed: maximum object size, whether object versioning exists, whether activation needs a payment card, and the `boto3` checksum settings in 4.1.

### 3.4 Access and security

- Create an R2 API token with **Object Read & Write**, limited to this one bucket. It gives an access key ID and a secret.
- Keys stay on the server. They never go into the Flutter app, notebook output, logs or git.
- Keep the bucket private: no public `r2.dev` URL and no public custom domain.
- Keep the path-traversal guard on downloaded keys.
- Use a separate bucket and token for development, so tests cannot damage production.

### 3.5 Environment variables

| Remove | Add |
|---|---|
| `SUPABASE_URL` | `R2_ACCOUNT_ID` |
| `SUPABASE_SERVICE_KEY` | `R2_ACCESS_KEY_ID` |
| `SUPABASE_BUCKET` | `R2_SECRET_ACCESS_KEY` |
| | `R2_BUCKET` (default `ebook-ai`) |

## 4. Code changes

### 4.1 New `r2_storage.py` (replaces `supabase_storage.py`)

Keep the same four function names, so `rag_service.py` and the test fake change very little. Import `boto3` lazily inside the client function.

```python
client = boto3.client(
    "s3",
    endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
    aws_access_key_id=...,
    aws_secret_access_key=...,
    region_name="auto",
    config=Config(
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=10,
        read_timeout=60,
        # verify against current docs: avoids checksum errors on S3-compatible services
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)
```

| Function | Implementation |
|---|---|
| `list_files(prefix)` | `list_objects_v2` paginator; add a trailing `/` to the prefix |
| `download_bytes(key)` | `get_object` |
| `download_file(key, dest)` (new) | `download_file`, streams to disk |
| `upload_file(key, local_path)` | `upload_file` (multipart for large files) |
| `delete_files(keys)` | `delete_objects`, in batches of 1000; check the returned errors |
| `read_data()` (new) | Return the parsed `data.json` and its ETag; `None` if missing |
| `update_data(mutate)` (new) | Conditional `put_object` with the retry rule in 3.2 |

### 4.2 `rag_service.py`

**`ensure_local_index()`**

1. Read `data.json` and compare `index.version` with the local `VERSION.json`.
2. If equal, stop.
3. Otherwise download the listed files into a temp dir and check each `sha256`.
4. Check `chroma.sqlite3` exists, write local `VERSION.json`, then `_activate()`.
5. On any failure, keep the local index (same behaviour as today).
6. If `data.json` or `index` is missing and there is no local index, report "no published index".

Rename `load_database_from_supabase` to `download_index`.

**`generate_and_store_database_task()`**

1. List `books/` and index one PDF at a time, as today.
2. Upload the index to `chroma_db/<version>/`.
3. **Last:** update `data.json` (`books` and `index`). This is the publish point.
4. Swap the local index.
5. Delete all older index folders: only the newest index is kept.

If anything before step 3 fails, `data.json` still points to the previous good index.

**Unchanged:** `warm_up`, retrieval, prompt, streaming, directory swap, both locks.

### 4.3 `main.py`

Existing endpoints keep the same contract. Only comments change.

Optional, only if the Flutter app needs the catalog or files (section 8):

- `GET /books` returns `data.json.books`, cached for a few minutes.
- `GET /books/{id}/url` returns a short-lived presigned URL, so downloads go straight from R2 and do not pass through the server.

Both expose the books, so decide on authentication first.

### 4.4 Other files

- `requirements.txt`: remove `supabase`, add `boto3`.
- `tests/test_index.py`: replace `FakeSB`. Add tests for checksum mismatch, `data.json` conflict retry, missing `data.json`, and "failed upload leaves `data.json` unchanged".
- `CLAUDE.md`, `docs/API.md`: update storage and env vars.
- `.claude/agents/security-reviewer.md`, `rag-pipeline-reviewer.md`: replace Supabase checks.
- `notebooks/ebook_rag_colab.ipynb`: port to R2 or delete.
- `.mcp.json`: remove the Supabase server.

## 5. Moving the data

One-off script `scripts/migrate_supabase_to_r2.py`, run locally while `supabase_storage.py` still exists.

1. Inventory the Supabase bucket: every path, size and total.
2. Copy `books/` and any images to R2 under the same keys.
3. Record a `sha256` for each file, then download from R2 and compare.
4. Create `data.json` with the book list and `"index": null`.
5. Do not copy `chroma_db/`. Rebuild it with the new code.

The script must be safe to run twice, must support a dry run, and must never delete anything from Supabase.

Afterwards, add books with the Cloudflare dashboard or `rclone`, then call `/generate-database/`.

## 6. Phases

| Phase | Work | Done when |
|---|---|---|
| 0 | Answer section 8; run the inventory | Total size and cost are acceptable |
| 1 | Create buckets (dev, prod) and tokens | A test upload and download works |
| 2 | Write `r2_storage.py` and tests | Tests pass without network |
| 3 | Run the data migration | Checksums match |
| 4 | Switch `rag_service.py`; publish the first index | A local run reaches `ready` and answers |
| 5 | Deploy with the new env vars on Render | Section 7 passes |
| 6 | After 2 stable weeks, remove Supabase code, env vars, keys and project | No `supabase` left outside history docs |

Do the code work on a branch from `dev`.

## 7. Acceptance checks

- `/health` reports `ready` with the new `index_version` and chunk count.
- The same test questions return the same books in `sources` as before.
- A server with an existing local index makes no R2 calls at startup or during queries.
- A server with an empty disk downloads the index and becomes ready.
- `/load-database/` returns `changed: false` when nothing changed.
- `/generate-database/` creates one new index folder and updates `data.json`.
- With R2 unreachable, a server with a local index still answers.
- Keys do not appear in logs or error responses.

**Rollback:** redeploy the previous commit with the Supabase env vars. This works until phase 6. Books added after the switch must be copied back.

## 8. Open questions

1. Does the Flutter app talk to Supabase directly (listing, downloading, login, tables)? If yes, plan that separately and do not delete the Supabase project.
2. What is the total size of the books and images?
3. Are the images book covers (assumed), or should OCR uploads now be saved?
4. Will books be added by hand (assumed) or through an admin upload endpoint?
