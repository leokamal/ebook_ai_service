"""The book list inside data.json."""
import posixpath
import re
from datetime import datetime, timezone


def _unique_id(file_name, taken):
    stem = posixpath.splitext(posixpath.basename(file_name))[0]
    base = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "book"
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


def sync_books(books, objects, indexed=None):
    """Make the book list match the PDFs in storage.

    books    current entries; fields edited by hand (title, cover, lang, ...) are kept.
    objects  r2_storage.list_objects("books") result.
    indexed  keys included in the search index, or None to leave the flags unchanged.
    """
    pdfs = {o["key"]: o for o in objects if o["key"].lower().endswith(".pdf")}
    result = [dict(b) for b in books if b.get("file_name") in pdfs]  # drop entries without a PDF
    known = {b["file_name"] for b in result}
    taken = {b.get("id") for b in result}
    for key in sorted(set(pdfs) - known):
        stem = posixpath.splitext(posixpath.basename(key))[0]
        entry = {
            "id": _unique_id(key, taken),
            "title": re.sub(r"[-_]+", " ", stem).strip(),
            "file_name": key,
            "cover": None,
            "lang": "en",
            "indexed": False,
            "added_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        taken.add(entry["id"])
        result.append(entry)
    for book in result:
        obj = pdfs[book["file_name"]]
        book.update(size_bytes=obj["size"], etag=obj["etag"])
        if indexed is not None:
            book["indexed"] = book["file_name"] in indexed
    return result
