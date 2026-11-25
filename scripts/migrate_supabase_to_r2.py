"""One-off copy from Supabase to Cloudflare R2. Never writes to or deletes from Supabase.

    pip install -r requirements.txt -r scripts/requirements-migration.txt
    python scripts/migrate_supabase_to_r2.py --dry-run   # needs only SUPABASE_* variables
    python scripts/migrate_supabase_to_r2.py             # also needs R2_* variables

Copies books/ and images/, then writes data.json from the public.books table.
chroma_db/ is not copied: rebuild it with POST /generate-database/. Safe to run again.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import catalog  # noqa: E402
import r2_storage as r2  # noqa: E402
import supabase_storage as source  # noqa: E402

FOLDERS = ("books", "images")
TABLE_PAGE = 1000


def read_book_rows():
    rows, start = [], 0
    while True:
        page = (source._client().table("books").select("*").order("id")
                .range(start, start + TABLE_PAGE - 1).execute().data)
        rows.extend(page)
        if len(page) < TABLE_PAGE:
            return rows
        start += TABLE_PAGE


def row_to_fields(row):
    """public.books row -> (storage key of the PDF, data.json fields)."""
    url = (row.get("url") or "").lstrip("/")
    key = url if url.startswith("books/") else f"books/{row['file_name']}"
    return key, {
        "legacy_id": row.get("id"),
        "title": row.get("title"),
        "cover": row.get("image_url"),
        "lang": row.get("lang") or "en",
        "added_at": row.get("created_at"),
    }


def merge_books(existing, rows, objects):
    """Table values fill new entries; entries already in data.json keep their own values."""
    present = {b.get("file_name") for b in existing}
    from_table = dict(row_to_fields(r) for r in rows)
    books = catalog.sync_books(existing, objects)
    for book in books:
        fields = from_table.get(book["file_name"])
        if not fields:
            continue
        for name, value in fields.items():
            if value is not None and (book["file_name"] not in present or book.get(name) is None):
                book[name] = value
    return books


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def copy_file(key, existing_keys):
    data = source.download_bytes(key)
    expected = sha256(data)
    if key in existing_keys and sha256(r2.download_bytes(key)) == expected:
        print(f"  unchanged  {key}")
        return len(data)
    with tempfile.TemporaryDirectory() as tmp:
        local_path = os.path.join(tmp, "object")
        with open(local_path, "wb") as f:
            f.write(data)
        r2.upload_file(key, local_path)
    if sha256(r2.download_bytes(key)) != expected:
        raise RuntimeError(f"Checksum mismatch after upload: {key}")
    print(f"  copied     {key} ({len(data):,} bytes)")
    return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="list what would be copied; write nothing")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    keys = [k for folder in FOLDERS for k in source.list_files(folder)
            if not k.endswith(".emptyFolderPlaceholder")]
    rows = read_book_rows()
    print(f"Supabase: {len(keys)} file(s), {len(rows)} row(s) in public.books")

    missing = []
    for row in rows:
        pdf_key, fields = row_to_fields(row)
        missing += [k for k in (pdf_key, fields["cover"]) if k and k not in keys]
    for key in missing:
        print(f"WARNING: the table references a file that is not in storage: {key}")

    if args.dry_run:
        for key in keys:
            print(f"  would copy {key}")
        print("Table rows:")
        print(json.dumps([row_to_fields(r)[1] for r in rows], ensure_ascii=False, indent=2))
        return

    existing_keys = {k for folder in FOLDERS for k in r2.list_files(folder)}
    total = sum(copy_file(key, existing_keys) for key in keys)
    print(f"Verified {len(keys)} file(s), {total:,} bytes")

    def mutate(data):
        data["books"] = merge_books(data.get("books", []), rows, r2.list_objects("books"))
        data.setdefault("index", None)

    data = r2.update_data(mutate)
    print(f"data.json written with {len(data['books'])} book(s)")


if __name__ == "__main__":
    main()
