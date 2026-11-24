import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import rag_service as rag
from security import verify_admin_token


# ---- admin token
def test_admin_disabled_without_env(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    with pytest.raises(HTTPException) as e:
        verify_admin_token("x")
    assert e.value.status_code == 503


def test_admin_rejects_wrong_or_missing(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    for bad in (None, "", "nope"):
        with pytest.raises(HTTPException) as e:
            verify_admin_token(bad)
        assert e.value.status_code == 403


def test_admin_accepts_correct(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    assert verify_admin_token("secret") is None


# ---- SSE format
def _collect(query="q", file_name=None):
    async def run():
        return [e async for e in rag.stream_query_events(query, file_name)]
    return asyncio.run(run())


def _parse(event):
    lines = event.strip().split("\n")
    return lines[0][len("event: "):], json.loads(lines[1][len("data: "):])


def test_sse_sources_tokens_done(monkeypatch):
    doc = SimpleNamespace(page_content="c", metadata={"file_name": "books/x.pdf"})
    monkeypatch.setattr(rag, "retrieve", lambda q, f=None: [doc])

    async def fake_stream(q, docs):
        for t in ("Hel", "lo"):
            yield t
    monkeypatch.setattr(rag, "astream_answer", fake_stream)

    events = [_parse(e) for e in _collect()]
    assert events == [
        ("sources", {"sources": ["books/x.pdf"]}),
        ("token", {"text": "Hel"}),
        ("token", {"text": "lo"}),
        ("done", {}),
    ]


def test_sse_not_found_skips_llm(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, f=None: [])
    monkeypatch.setattr(rag, "astream_answer", lambda *a: (_ for _ in ()).throw(AssertionError("LLM called")))
    events = [_parse(e) for e in _collect()]
    assert events[0] == ("sources", {"sources": []})
    assert events[1] == ("token", {"text": rag.NOT_FOUND_ANSWER})
    assert events[-1] == ("done", {})


def test_sse_error_event(monkeypatch):
    def boom(q, f=None):
        raise RuntimeError("secret internals")
    monkeypatch.setattr(rag, "retrieve", boom)
    events = [_parse(e) for e in _collect()]
    assert events == [("error", {"detail": "Failed to generate the answer"})]


def test_file_name_filter_passed(monkeypatch):
    seen = {}

    class Store:
        def similarity_search_with_relevance_scores(self, q, **kw):
            seen.update(kw)
            return []
    monkeypatch.setattr(rag, "_state", {"store": Store(), "version": None})
    rag.retrieve("q", "books/x.pdf")
    assert seen["filter"] == {"file_name": "books/x.pdf"}
