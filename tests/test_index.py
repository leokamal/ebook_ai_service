import json
import os

import pytest

import rag_service as rag


class FakeSB:
    """Stand-in for supabase_storage."""
    def __init__(self, files):
        self.files = files            # {"chroma_db/VERSION.json": bytes, ...}
        self.downloads = []

    def download_bytes(self, name):
        self.downloads.append(name)
        if name not in self.files:
            raise FileNotFoundError(name)
        return self.files[name]

    def list_files(self, prefix):
        return [n for n in self.files if n.startswith(prefix.strip("/") + "/")]


def _remote(version="v2"):
    return {
        "chroma_db/VERSION.json": json.dumps({"version": version}).encode(),
        "chroma_db/chroma.sqlite3": b"sqlite",
        "chroma_db/seg/data.bin": b"bin",
    }


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


def test_same_version_skips_download(env, monkeypatch):
    chroma, activated = env
    _local(chroma, "v2")
    fake = FakeSB(_remote("v2"))
    monkeypatch.setattr(rag, "sb", fake)
    assert rag.ensure_local_index() is False
    assert fake.downloads == ["chroma_db/VERSION.json"]   # only VERSION.json fetched
    assert activated == []


def test_different_version_downloads_and_swaps(env, monkeypatch):
    chroma, activated = env
    _local(chroma, "v1")
    monkeypatch.setattr(rag, "sb", FakeSB(_remote("v2")))
    assert rag.ensure_local_index() is True
    assert rag.read_local_version()["version"] == "v2"
    assert (chroma / "seg" / "data.bin").read_bytes() == b"bin"
    assert not os.path.exists(str(chroma) + ".old")


def test_missing_local_downloads(env, monkeypatch):
    chroma, _ = env
    monkeypatch.setattr(rag, "sb", FakeSB(_remote("v1")))
    assert rag.ensure_local_index() is True
    assert rag.has_local_index()


def test_supabase_unreachable_uses_local(env, monkeypatch):
    chroma, activated = env
    _local(chroma, "v1")
    monkeypatch.setattr(rag, "sb", FakeSB({}))   # VERSION.json missing -> error
    assert rag.ensure_local_index() is False
    assert rag.read_local_version()["version"] == "v1"


def test_incomplete_download_keeps_old(env, monkeypatch):
    chroma, _ = env
    _local(chroma, "v1")
    files = _remote("v2")
    del files["chroma_db/chroma.sqlite3"]
    monkeypatch.setattr(rag, "sb", FakeSB(files))
    assert rag.ensure_local_index() is False
    assert rag.read_local_version()["version"] == "v1"


def test_path_traversal_is_skipped(tmp_path, monkeypatch):
    files = _remote("v2")
    files["chroma_db/../evil.txt"] = b"x"
    monkeypatch.setattr(rag, "sb", FakeSB(files))
    dest = tmp_path / "dest"
    rag.load_database_from_supabase(str(dest))
    assert not (tmp_path / "evil.txt").exists()
    assert (dest / "chroma.sqlite3").exists()


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
