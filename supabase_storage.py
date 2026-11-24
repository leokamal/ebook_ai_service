"""Thin wrapper around Supabase Storage (replaces the former Firebase Storage usage).

Required environment variables:
  SUPABASE_URL          e.g. https://<project-ref>.supabase.co
  SUPABASE_SERVICE_KEY  service_role key (server-side only, never ship to clients)
Optional:
  SUPABASE_BUCKET       storage bucket name (default: "ebook-ai")
"""
import os
from functools import lru_cache

PAGE_SIZE = 100


def bucket_name() -> str:
    return os.environ.get("SUPABASE_BUCKET", "ebook-ai")


@lru_cache(maxsize=1)
def _client():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key)


def _bucket():
    return _client().storage.from_(bucket_name())


def list_files(prefix: str) -> list[str]:
    """Recursively list file paths under `prefix` (Supabase list() is one level only)."""
    prefix = prefix.strip("/")
    results: list[str] = []
    offset = 0
    while True:
        items = _bucket().list(prefix, {"limit": PAGE_SIZE, "offset": offset})
        for item in items:
            path = f"{prefix}/{item['name']}" if prefix else item["name"]
            if item.get("id") is None:  # folder placeholder
                results.extend(list_files(path))
            else:
                results.append(path)
        if len(items) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return results


def download_bytes(path: str) -> bytes:
    return _bucket().download(path)


def upload_file(path: str, local_path: str) -> None:
    with open(local_path, "rb") as f:
        _bucket().upload(
            path,
            f.read(),
            {"content-type": "application/octet-stream", "upsert": "true"},
        )


def delete_files(paths: list[str]) -> None:
    if paths:
        _bucket().remove(paths)
