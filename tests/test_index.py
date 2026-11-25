import hashlib
import json
import os
import sys
import types

import pytest

import catalog
import rag_service as rag


class FakeStorage:
    """Stand-in for r2_storage."""
    def __init__(self, files=None, data=None):
        self.files = dict(files or {})  # {key: bytes}
        self.data = data
        self.downloads = []
        self.deleted = []
        self.fail_upload = False
        self.unreachable = False

    def read_data(self):
        if self.unreachable:
            raise ConnectionError("down")
        return (json.loads(json.dumps(self.data)), "etag") if self.data else (None, None)

    def update_data(self, mutate):
        data = self.read_data()[0] or {"schema_version": 1, "books": [], "index": None}
        mutate(data)
        self.data = data
        return data

    def download_file(self, key, local_path):
        self.downloads.append(key)
        with open(local_path, "wb") as f:
            f.write(self.files[key])

    def download_bytes(self, key):
        return self.files[key]

    def upload_file(self, key, local_path):
        if self.fail_upload:
            raise ConnectionError("upload failed")
        with open(local_path, "rb") as f:
            self.files[key] = f.read()

    def list_objects(self, prefix):
        return [{"key": k, "size": len(v), "etag": "e"} for k, v in sorted(self.files.items())
                if k.startswith(prefix.strip("/") + "/")]

    def list_files(self, prefix):
        return [o["key"] for o in self.list_objects(prefix)]

    def delete_files(self, keys):
        self.deleted += keys
        for k in keys:
            del self.files[k]


def _remote(version="v2", **overrides):
    content = {"chroma.sqlite3": b"sqlite", "seg/data.bin": b"bin"}
    prefix = f"chroma_db/{version}"
    index = {
        "version": version,
        "prefix": prefix,
        "files": [{"path": p, "sha256": hashlib.sha256(b).hexdigest()} for p, b in content.items()],
        "chunks": 2,
        "embedding_model": "m",
        **overrides,
    }
    books = [{"file_name": "books/a.pdf", "indexed": True}]
    return FakeStorage({f"{prefix}/{p}": b for p, b in content.items()},
                       {"schema_version": 1, "books": books, "index": index})


def _local(path, version):
    path.mkdir(exist_ok=True)
    (path / "chroma.sqlite3").write_bytes(b"old")
    (path / "VERSION.json").write_text(json.dumps({"version": version}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    chroma = tmp_path / "chroma"
    monkeypatch.setattr(rag, "CHROMA_DIR", str(chroma))
    activated = []
    monkeypatch.setattr(rag, "_activate", lambda new_dir=None: (
        rag.swap_directory(new_dir, rag.CHROMA_DIR) if new_dir else None, activated.append(new_dir)))
    return chroma, activated


def _use(monkeypatch, fake):
    monkeypatch.setattr(rag, "storage", fake)
    return fake


# ---------------------------------------------------------------------------- downloading
def test_same_version_skips_download(env, monkeypatch):
    chroma, activated = env
    _local(chroma, "v2")
    fake = _use(monkeypatch, _remote("v2"))
    assert rag.ensure_local_index() is False
    assert fake.downloads == []
    assert activated == []


def test_different_version_downloads_and_swaps(env, monkeypatch):
    chroma, _ = env
    _local(chroma, "v1")
    _use(monkeypatch, _remote("v2"))
    assert rag.ensure_local_index() is True
    assert rag.read_local_version() == {
        "version": "v2", "files": ["books/a.pdf"], "chunks": 2, "embedding_model": "m"}
    assert (chroma / "seg" / "data.bin").read_bytes() == b"bin"
    assert not os.path.exists(str(chroma) + ".old")


def test_missing_local_downloads(env, monkeypatch):
    _use(monkeypatch, _remote("v1"))
    assert rag.ensure_local_index() is True
    assert rag.has_local_index()


def test_unreachable_uses_local(env, monkeypatch):
    chroma, _ = env
    _local(chroma, "v1")
    _use(monkeypatch, _remote("v2")).unreachable = True
    assert rag.ensure_local_index() is False
    assert rag.read_local_version()["version"] == "v1"


@pytest.mark.parametrize("data", [None, {"books": [], "index": None}])
def test_no_published_index(env, monkeypatch, data):
    chroma, _ = env
    _use(monkeypatch, FakeStorage(data=data))
    with pytest.raises(RuntimeError, match="No published index"):
        rag.ensure_local_index()
    _local(chroma, "v1")
    assert rag.ensure_local_index() is False


def test_incomplete_download_keeps_old(env, monkeypatch):
    chroma, _ = env
    _local(chroma, "v1")
    fake = _use(monkeypatch, _remote("v2"))
    del fake.files["chroma_db/v2/chroma.sqlite3"]
    assert rag.ensure_local_index() is False
    assert rag.read_local_version()["version"] == "v1"


def test_checksum_mismatch_keeps_old(env, monkeypatch):
    chroma, activated = env
    _local(chroma, "v1")
    _use(monkeypatch, _remote("v2")).files["chroma_db/v2/seg/data.bin"] = b"corrupted"
    assert rag.ensure_local_index() is False
    assert rag.read_local_version()["version"] == "v1"
    assert activated == []


@pytest.mark.parametrize("path", ["../evil.txt", "/evil.txt", "a/../../evil.txt"])
def test_path_traversal_is_rejected(tmp_path, monkeypatch, path):
    fake = _use(monkeypatch, FakeStorage({f"chroma_db/v/{path}": b"x"}))
    index = {"prefix": "chroma_db/v",
             "files": [{"path": path, "sha256": hashlib.sha256(b"x").hexdigest()}]}
    with pytest.raises(ValueError):
        rag.download_index(index, str(tmp_path / "dest"))
    assert fake.downloads == []
    assert not (tmp_path / "evil.txt").exists()


# ----------------------------------------------------------------------------- publishing
@pytest.fixture
def fake_libs(monkeypatch):
    """Replace the heavy libraries used while generating the index."""
    class Chroma:
        def __init__(self, embedding_function, persist_directory):
            self.directory = persist_directory

        def add_documents(self, docs):
            with open(os.path.join(self.directory, "chroma.sqlite3"), "wb") as f:
                f.write(b"index")

    class Splitter:
        def __init__(self, **_):
            pass

        def split_text(self, text):
            return [text]

    class PdfReader:
        def __init__(self, stream):
            text = stream.read().decode()
            self.pages = [types.SimpleNamespace(extract_text=lambda: text)]

    for module, attrs in {
        "PyPDF2": {"PdfReader": PdfReader},
        "langchain_chroma": {"Chroma": Chroma},
        "langchain_core": {},
        "langchain_core.documents": {"Document": lambda **kw: kw},
        "langchain_text_splitters": {"RecursiveCharacterTextSplitter": Splitter},
    }.items():
        fake = types.ModuleType(module)
        fake.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, module, fake)
    monkeypatch.setattr(rag, "get_embeddings_model", lambda: None)


def _published():
    old = {"version": "20260102T000000Z"}
    return FakeStorage(
        {"books/a.pdf": b"hello",
         "chroma_db/20260101T000000Z/chroma.sqlite3": b"older",
         "chroma_db/20260102T000000Z/chroma.sqlite3": b"previous",
         "chroma_db/29990101T000000Z/chroma.sqlite3": b"other server uploading"},
        {"schema_version": 1, "index": old,
         "books": [{"id": "a", "title": "Hand edited", "file_name": "books/a.pdf", "cover": "images/a.png"},
                   {"id": "gone", "file_name": "books/gone.pdf"}]})


def test_generate_publishes_and_cleans_up(env, monkeypatch, fake_libs):
    chroma, _ = env
    fake = _use(monkeypatch, _published())
    rag.generate_and_store_database_task()

    index = fake.data["index"]
    assert index["prefix"] == f"chroma_db/{index['version']}"
    assert [f["path"] for f in index["files"]] == ["chroma.sqlite3"]
    assert f"{index['prefix']}/chroma.sqlite3" in fake.files
    [book] = fake.data["books"]
    assert (book["title"], book["cover"], book["indexed"]) == ("Hand edited", "images/a.png", True)
    # only the new index remains (plus a newer folder another server is still uploading)
    assert fake.deleted == ["chroma_db/20260101T000000Z/chroma.sqlite3",
                            "chroma_db/20260102T000000Z/chroma.sqlite3"]
    assert fake.list_files("chroma_db") == [
        f"{index['prefix']}/chroma.sqlite3", "chroma_db/29990101T000000Z/chroma.sqlite3"]
    assert rag.read_local_version()["version"] == index["version"]


def test_failed_upload_leaves_data_unchanged(env, monkeypatch, fake_libs):
    chroma, activated = env
    fake = _use(monkeypatch, _published())
    fake.fail_upload = True
    before = json.dumps(fake.data)
    with pytest.raises(ConnectionError):
        rag.generate_and_store_database_task()
    assert json.dumps(fake.data) == before
    assert fake.deleted == [] and activated == []
    assert not list(chroma.parent.glob(".chroma_new_*"))


# -------------------------------------------------------------------------------- catalog
def test_sync_books_adds_new_pdfs_with_unique_ids():
    objects = [{"key": k, "size": 1, "etag": "e"}
               for k in ("books/My_Book.pdf", "books/my-book.pdf", "books/notes.txt")]
    books = catalog.sync_books([], objects)
    assert [b["id"] for b in books] == ["my-book", "my-book-2"]
    assert books[0]["title"] == "My Book"


# ------------------------------------------------------------------------- swap directory
def test_swap_directory_atomic(tmp_path):
    target, new = tmp_path / "t", tmp_path / "n"
    target.mkdir(); (target / "a").write_text("old")
    new.mkdir(); (new / "b").write_text("new")
    rag.swap_directory(str(new), str(target))
    assert (target / "b").read_text() == "new" and not (target / "a").exists()
    assert not new.exists() and not (tmp_path / "t.old").exists()


def test_swap_rolls_back_on_failure(tmp_path):
    target = tmp_path / "t"
    target.mkdir(); (target / "a").write_text("old")
    with pytest.raises(OSError):
        rag.swap_directory(str(tmp_path / "missing"), str(target))
    assert (target / "a").read_text() == "old"
