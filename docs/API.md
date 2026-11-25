# Ebook AI API (for the Flutter client)

Base URL: `https://ebook-ai-service.onrender.com`

## GET /health
```json
{"status": "ready", "index_version": "20261124T031500Z", "chunks": 812}
```
`status` is `starting` | `ready` | `error` (an `error` field is added on failure). Use it as the Render health check and for keep-alive pings.

## POST /query  (Server-Sent Events)
Request (`Content-Type: application/json`):
```json
{"query": "What is a StatefulWidget?", "file_name": "books/flutter-succinctly.pdf"}
```
- `query`: 1-1000 characters, required.
- `file_name`: optional; restricts the search to one book (the storage path, as listed in `sources`).

Response: `text/event-stream`. Events, in order:
```
event: sources
data: {"sources": ["books/flutter-succinctly.pdf"]}

event: token
data: {"text": "A StatefulWidget is "}

event: token
data: {"text": "a widget that ..."}

event: done
data: {}
```
- `sources` is sent right after retrieval, before any token.
- If no passage matches, one `token` with "I couldn't find this information in the books." is sent (no LLM call), then `done`.
- On failure: `event: error` with `data: {"detail": "..."}`, and the stream ends.

Dart sketch: send the POST with `http.Client().send(...)`, then split `response.stream.transform(utf8.decoder)` on blank lines (`\n\n`); each block has an `event:` and a `data:` line.

## 503 while warming up
If the index is not loaded yet, `/query` and `/query-database/` wait up to 30 s, then return:
```json
{"detail": "Service warming up, retry shortly"}
```
Retry after 2-5 s. If loading failed: `{"detail": "Index unavailable: ..."}`.

## GET /query-database/?query=...
Non-streaming: `{"answer": "...", "context": [{"page_content": "retrieved chunk", "metadata": {"file_name": "books/x.pdf"}}, ...]}`. The streaming `sources` event also carries `context` in the same shape.

## Admin (not for the app)
`POST /generate-database/` and `POST /load-database/` need header `X-Admin-Token: <ADMIN_TOKEN>`; without it: 403 (503 if `ADMIN_TOKEN` is not configured on the server).
