# Plan: migrate from Supabase to a single GitHub repository

Status: superseded by `migrate_supabase_to_r2.md`. Do not implement.

## 1. Goal

- One new GitHub repository holds all stored files (PDFs, images).
- One `data.json` file in that repository is the only database.
- Supabase is removed from the service.

## 2. Where we are today

- The service uses **Supabase Storage only**. The code makes no Postgres table calls.
- `books/*.pdf` are the source books.
- `chroma_db/` is the published search index; `chroma_db/VERSION.json` says which version it is.
- `/upload` runs OCR in memory and stores nothing.
- All storage access goes through `supabase_storage.py` and is used by `rag_service.py`.
- Queries use the local Chroma copy only. This stays true after the migration.

## 3. Target design

### 3.1 Repository

New **private** repository, separate from the service code. Suggested name: `ebook-ai-storage`.

```
ebook-ai-storage/
├── data.json          # the database
├── books/             # PDFs (same paths as today)
│   └── flutter-succinctly.pdf
├── images/
│   └── covers/
└── README.md          # initial commit; the repo must not be empty

GitHub Releases
└── index-<version>    # asset: chroma_db.zip
```

**Keep `books/<name>.pdf` paths unchanged.** They are stored as `file_name` in chunk metadata and the Flutter app sends them back to `/query`.

**Why the Chroma index goes in a Release asset, not in a commit:** the index is binary and is rebuilt in full. Every commit of it would stay in git history forever and the repository would keep growing. A Release asset is in the same repository, can be deleted, and does not add to history.

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
      "sha": "<git blob sha>",
      "indexed": true,
      "added_at": "2026-11-25T10:00:00Z"
    }
  ],
  "index": {
    "version": "20261125T100000Z",
    "release_tag": "index-20261125T100000Z",
    "asset_name": "chroma_db.zip",
    "sha256": "<hex>",
    "chunks": 812,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

Rules:

- `data.json.index` replaces the remote `VERSION.json`. A local `VERSION.json` is still written in `CHROMA_DIR`, with the same fields as today, so `/health` does not change.
- Only the service writes `index`, `size_bytes`, `sha`, `indexed`. Hand-edited fields (`title`, `cover`) must be preserved on rewrite.
- A PDF found in `books/` with no entry gets one automatically (title taken from the file name).
- An entry whose PDF no longer exists is removed.
- Never put secrets or personal user data in it.

**Limit of this design:** one JSON file suits a small catalog with one writer. It has no queries, no transactions across writers, and every change rewrites the whole file. If user accounts, favourites or reading progress are needed later, use a real database.

### 3.3 GitHub limits that shape the design

Check these against current GitHub documentation before implementing.

| Limit | Effect |
|---|---|
| A file in git cannot exceed 100 MB (warning above 50 MB) | Larger PDFs need a Release asset or compression |
| Release asset: up to 2 GB each | Fine for the index zip |
| Contents API returns inline content only up to 1 MB | Always request the raw media type |
| 5,000 authenticated API requests per hour, plus secondary limits on writes | Fine at this scale; handle 403/429 with backoff |
| Repositories are best kept around 1 GB, strongly under 5 GB | Watch total PDF size |
| GitHub is not meant to serve as a file CDN | Do not serve heavy public traffic from it |

Do not use Git LFS: it has separate storage and bandwidth quotas.

### 3.4 Access and security

- Fine-grained personal access token, limited to this one repository, permission **Contents: read and write** (covers releases).
- The token lives only on the server. It never goes into the Flutter app, the notebook output, logs or the repo.
- Tokens expire. Record the expiry date and set a reminder.
- Keep the repository private: a public one would publish the books to everyone.
- Keep the path-traversal guard, and apply it to zip entries when extracting the index (zip-slip).
- Do not forward the `Authorization` header to the redirected asset download host.

### 3.5 Environment variables

| Remove | Add |
|---|---|
| `SUPABASE_URL` | `GITHUB_REPO` (`owner/name`) |
| `SUPABASE_SERVICE_KEY` | `GITHUB_STORAGE_TOKEN` |
| `SUPABASE_BUCKET` | `GITHUB_BRANCH` (default `main`) |

## 4. Code changes

### 4.1 New `github_storage.py` (replaces `supabase_storage.py`)

Use `httpx` and add it to `requirements.txt` explicitly. Keep the import lazy.

| Function | GitHub API |
|---|---|
| `head_sha()` | Get the branch ref |
| `list_files(prefix, ref)` | Git Trees, recursive; fail if the response is `truncated` |
| `download_bytes(path, ref)` | Contents API, raw media type |
| `commit_files(changes, deletes, message)` | Git Data API: blobs, tree, commit, update ref (no force) |
| `read_data()` | Download `data.json`; return it with the commit sha |
| `update_data(mutate)` | Read, apply `mutate`, commit; on conflict re-read and retry (max 3) |
| `create_release(tag)` / `upload_asset()` | Releases API |
| `download_asset(tag, name, dest)` | Releases API, streamed to disk |
| `delete_old_releases(keep=2)` | Delete releases and their tags |

All calls need timeouts and clear errors for 401, 404, 409/422 and 403/429.

### 4.2 `rag_service.py`

**`ensure_local_index()`**

1. Read `data.json` and compare `index.version` with the local `VERSION.json`.
2. If equal, stop. Only `data.json` was fetched.
3. Otherwise download the zip to a temp dir, check `sha256`, extract with the path guard.
4. Check `chroma.sqlite3` exists, write local `VERSION.json`, swap with `_activate()`.
5. On any failure, keep the local index (same behaviour as today).

Rename `load_database_from_supabase` to `download_index`.

**`generate_and_store_database_task()`**

1. Take one commit sha and list and download all PDFs at that sha.
2. Build the index in a temp dir, one PDF at a time, as today.
3. Zip it and compute `sha256`.
4. Create release `index-<version>` and upload the asset.
5. **Last:** update `data.json` (`books` and `index`). This is the publish point.
6. Swap the local index, then delete old releases (keep 2).

If any step before 5 fails, `data.json` still points to the previous good index.

**Unchanged:** `warm_up`, retrieval, prompt, streaming, directory swap, both locks.

### 4.3 `main.py`

Existing endpoints keep the same contract. Update comments that mention Supabase.

Optional, only if the Flutter app needs the catalog or the files (see section 8):

- `GET /books` returns `data.json.books`, cached for a few minutes.
- `GET /files/{path}` streams a file, limited to `books/` and `images/`.

These need their own decision on authentication, because they expose the books.

### 4.4 Other files

- `tests/test_index.py`: replace `FakeSB` with a GitHub fake. Add tests for checksum mismatch, zip-slip, `data.json` conflict retry, and "failed upload leaves `data.json` unchanged".
- `requirements.txt`: remove `supabase`, add `httpx`.
- `CLAUDE.md`, `docs/API.md`: update storage and env vars.
- `.claude/agents/security-reviewer.md`, `rag-pipeline-reviewer.md`: replace Supabase checks.
- `notebooks/ebook_rag_colab.ipynb`: port to the GitHub helpers or delete.
- `.mcp.json`: remove the Supabase server.
- `docs/features/avoid_download_chromadb_from_supabase.md`: keep as history.

## 5. Moving the data

One-off script `scripts/migrate_supabase_to_github.py`, run locally while `supabase_storage.py` still exists.

1. **Inventory** the bucket: every path, size and total. Flag files over 50 MB and 100 MB.
2. Download `books/` and any images to a temp folder outside both repositories.
3. Push them to the new repository with plain `git` (simpler than the API for a bulk upload).
4. Create `data.json` from the file list, with `"index": null`.
5. **Verify** file count, sizes and checksums against the inventory.
6. Do not copy `chroma_db/`. Rebuild it with the new code.

The script must be safe to run twice and must never delete anything from Supabase.

While `index` is `null`, `ensure_local_index()` must report "no published index" instead of crashing.

## 6. Phases

| Phase | Work | Done when |
|---|---|---|
| 0 | Answer section 8; run the inventory | Sizes fit the limits |
| 1 | Create the repository and token | Token can read and write that repo only |
| 2 | Write `github_storage.py` and tests | Tests pass without network |
| 3 | Run the data migration | Verification matches |
| 4 | Switch `rag_service.py`; publish the first index | A local run reaches `ready` and answers |
| 5 | Deploy: set new env vars on Render | Checks in section 7 pass |
| 6 | After 2 stable weeks, remove Supabase code, env vars, key and project | No `supabase` left outside history docs |

Do phases 2 and 4 on a branch from `dev`.

## 7. Acceptance checks

- `/health` reports `ready` with the new `index_version` and chunk count.
- The same test questions return the same books in `sources` as before.
- Restart with a persistent disk makes no GitHub calls at startup (same as today).
- A fresh server with an empty disk downloads the index and becomes ready.
- `/load-database/` returns `changed: false` when nothing changed.
- `/generate-database/` creates one release and one `data.json` commit.
- With GitHub unreachable, a server with a local index still answers.
- A query makes no GitHub calls.
- The token does not appear in logs or error responses.

**Rollback:** redeploy the previous commit with the Supabase env vars. This works until phase 6, because Supabase data is not touched. Books added after the switch would need to be copied back.

## 8. Open questions

1. Does the Flutter app talk to Supabase directly (listing, downloading, login, tables)? If yes, those parts need their own plan, and this one must not remove the Supabase project.
2. How large is the biggest PDF, and the total?
3. What are the images: book covers, or should OCR uploads now be saved? This plan assumes covers.
4. How are new books added: `git push` (assumed) or an admin upload endpoint?
5. Do you have the right to store these books on GitHub?
