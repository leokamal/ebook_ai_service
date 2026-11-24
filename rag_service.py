"""RAG service: Chroma index kept locally (CHROMA_DIR) and published to Supabase Storage.

Flow
  startup  -> warm_up(): ensure_local_index() (downloads only if VERSION.json changed),
              load embeddings + open Chroma + dummy search, status -> "ready".
  query    -> retrieve() (local Chroma only, zero Supabase calls) -> Gemini (streamed or not).
  publish  -> generate_and_store_database_task(): build in a temp dir, upload files with
              VERSION.json LAST, delete stale remote files, swap local dir + in-memory state.

NOTE: heavy imports (torch/sentence-transformers, chroma, gemini, PyPDF2) are lazy, inside
functions, so the web server binds its port quickly. The pysqlite3 -> sqlite3 swap is done in
main.py before anything imports chromadb. GOOGLE_API_KEY must be set in the environment.
"""
import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache

import supabase_storage as sb

logger = logging.getLogger("uvicorn.error")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.5-flash"
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db_local")
VERSION_FILE = "VERSION.json"
REMOTE_PREFIX = "chroma_db"
TOP_K = 3
SCORE_THRESHOLD = 0.4
NOT_FOUND_ANSWER = "I couldn't find this information in the books."

PROMPT = """
You are a highly skilled assistant who excels in rewriting, summarizing, and generating relevant information based on a given context.
Your task is to rewrite and summarize the context so that it directly addresses the question.
Ensure that the rewritten context is relevant, concise, and directly related to the user's question.
If the data (context) doesn't exist, say so.

**Context:** {context}

**Question:** {input}

**Rewritten and Summarized Context:**
"""

# _update_lock serializes long operations (download / generate). _lock only guards the
# short in-memory state swap, so queries never wait on a download.
_update_lock = threading.Lock()
_lock = threading.Lock()
_state = None  # {"store": Chroma, "version": dict | None} - replaced atomically
_status = {"status": "starting", "error": None}
_finished = threading.Event()  # set once warm-up ends (ready or error)


# --------------------------------------------------------------------------- models
@lru_cache(maxsize=1)
def get_embeddings_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


@lru_cache(maxsize=1)
def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=LLM_MODEL)


# ------------------------------------------------------------------ local index state
def _version_path(directory=None):
    return os.path.join(directory or CHROMA_DIR, VERSION_FILE)


def read_local_version(directory=None):
    try:
        with open(_version_path(directory)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def has_local_index(directory=None):
    return os.path.exists(os.path.join(directory or CHROMA_DIR, "chroma.sqlite3"))


def _replace_contents(new_dir, target):
    """Empty `target` (keeping the directory itself) and move `new_dir`'s contents into it.
    shutil.move copies across filesystems when a plain rename isn't possible."""
    os.makedirs(target, exist_ok=True)
    for name in os.listdir(target):
        p = os.path.join(target, name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            os.remove(p)
    for name in os.listdir(new_dir):
        shutil.move(os.path.join(new_dir, name), os.path.join(target, name))
    shutil.rmtree(new_dir, ignore_errors=True)


def swap_directory(new_dir, target):
    """Atomically replace `target` with `new_dir` (rename old -> .old, new -> target, delete .old)."""
    old = target + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.ismount(target):
        # target is a mount point (e.g. a Render persistent disk): it can't be renamed,
        # so replace its contents in place instead.
        _replace_contents(new_dir, target)
        return
    if os.path.exists(target):
        try:
            os.rename(target, old)
        except OSError:
            _replace_contents(new_dir, target)
            return
    try:
        os.rename(new_dir, target)
    except Exception:
        if os.path.exists(old) and not os.path.exists(target):
            os.rename(old, target)  # roll back
        raise
    shutil.rmtree(old, ignore_errors=True)


def _clear_chroma_cache():
    """Chroma caches clients per path; drop them so a swapped directory is re-opened."""
    for mod in ("chromadb.api.shared_system_client", "chromadb.api.client"):
        try:
            __import__(mod, fromlist=["SharedSystemClient"]).SharedSystemClient.clear_system_cache()
            return
        except Exception:
            continue


def _open_store():
    from langchain_chroma import Chroma
    return Chroma(embedding_function=get_embeddings_model(), persist_directory=CHROMA_DIR)


def _activate(new_dir=None):
    """Swap `new_dir` into CHROMA_DIR (if given) and replace the in-memory state atomically."""
    global _state
    with _lock:
        if new_dir:
            swap_directory(new_dir, CHROMA_DIR)
        _clear_chroma_cache()
        _state = {"store": _open_store(), "version": read_local_version()}
        _status.update(status="ready", error=None)
    _finished.set()


def _get_state():
    with _lock:
        st = _state
    if st is None:
        raise RuntimeError("Index not loaded yet")
    return st


# ----------------------------------------------------------------- download from Supabase
def load_database_from_supabase(dest_dir):
    """Download every object under chroma_db/ into `dest_dir` (path-traversal guarded)."""
    prefix = f"{REMOTE_PREFIX}/"
    base = os.path.realpath(dest_dir)
    os.makedirs(base, exist_ok=True)
    for name in sb.list_files(REMOTE_PREFIX):
        rel_path = name[len(prefix):]
        if not rel_path:
            continue
        parts = rel_path.split("/")
        if any(p in ("", ".", "..") for p in parts):
            logger.warning("Skipping suspicious object name %s", name)
            continue
        local_path = os.path.realpath(os.path.join(base, *parts))
        if os.path.commonpath([base, local_path]) != base:
            logger.warning("Skipping object outside target dir: %s", name)
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(sb.download_bytes(name))
        logger.info("Downloaded %s", name)


def ensure_local_index() -> bool:
    """Make CHROMA_DIR match the published index. Returns True if a new index was installed.

    Only VERSION.json is fetched when nothing changed. Falls back to the local copy if
    Supabase is unreachable.
    """
    with _update_lock:
        local_v = read_local_version()
        try:
            remote_v = json.loads(sb.download_bytes(f"{REMOTE_PREFIX}/{VERSION_FILE}"))
        except Exception as e:
            if has_local_index():
                logger.warning("Supabase unreachable (%s); using local index.", e)
                return False
            raise RuntimeError(f"No local index and remote VERSION.json unavailable: {e}") from e

        if local_v and local_v.get("version") == remote_v.get("version") and has_local_index():
            logger.info("Local index is up to date (version %s); no download.", local_v.get("version"))
            return False

        parent = os.path.dirname(os.path.abspath(CHROMA_DIR)) or "."
        os.makedirs(parent, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=".chroma_new_", dir=parent)  # same filesystem as CHROMA_DIR
        try:
            load_database_from_supabase(tmp)
            if not (has_local_index(tmp) and os.path.exists(_version_path(tmp))):
                raise RuntimeError("Downloaded index is incomplete (chroma.sqlite3 / VERSION.json missing)")
        except Exception as e:
            shutil.rmtree(tmp, ignore_errors=True)
            if has_local_index():
                logger.warning("Index download failed (%s); keeping local index.", e)
                return False
            raise
        _activate(tmp)
        logger.info("Installed index version %s", remote_v.get("version"))
        return True


# ----------------------------------------------------------------------------- warm-up
def warm_up():
    try:
        ensure_local_index()
        with _lock:
            loaded = _state is not None
        if not loaded:
            _activate()
        try:
            _get_state()["store"].similarity_search("warm up", k=1)  # load everything into memory
        except Exception as e:
            logger.warning("Warm-up search failed: %s", e)
        get_llm()
        _status.update(status="ready", error=None)
        logger.info("RAG service ready")
    except Exception as e:
        logger.exception("Warm-up failed")
        _status.update(status="error", error=str(e))
    finally:
        _finished.set()


def start_warmup_thread():
    t = threading.Thread(target=warm_up, name="rag-warmup", daemon=True)
    t.start()
    return t


def wait_ready(timeout: float = 30) -> bool:
    _finished.wait(timeout)
    return _status["status"] == "ready"


def get_health():
    with _lock:
        st = _state
    version = (st or {}).get("version") or {}
    return {
        "status": _status["status"],
        "index_version": version.get("version"),
        "chunks": version.get("chunks"),
        **({"error": _status["error"]} if _status["error"] else {}),
    }


# ------------------------------------------------------------------------------ querying
def retrieve(query, file_name=None):
    """Similarity search on the local Chroma index; keeps chunks above SCORE_THRESHOLD."""
    kwargs = {"k": TOP_K}
    if file_name:
        kwargs["filter"] = {"file_name": file_name}
    results = _get_state()["store"].similarity_search_with_relevance_scores(query, **kwargs)
    return [doc for doc, score in results if score >= SCORE_THRESHOLD]


def _build_prompt(query, docs):
    return PROMPT.format(context="\n\n".join(d.page_content for d in docs), input=query)


def _chunk_text(chunk):
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return ""


async def astream_answer(query, docs):
    async for chunk in get_llm().astream(_build_prompt(query, docs)):
        text = _chunk_text(chunk)
        if text:
            yield text


def query_database(query, file_name=None):
    """Non-streaming answer (used by GET /query-database/)."""
    t0 = time.perf_counter()
    docs = retrieve(query, file_name)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    answer = _chunk_text(get_llm().invoke(_build_prompt(query, docs))) if docs else NOT_FOUND_ANSWER
    logger.info("query retrieve_ms=%.0f total_ms=%.0f", retrieve_ms, (time.perf_counter() - t0) * 1000)
    return answer


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_query_events(query, file_name=None):
    """Yield Server-Sent Events: sources, token*, done (or error)."""
    t0 = time.perf_counter()
    try:
        docs = await asyncio.to_thread(retrieve, query, file_name)
        retrieve_ms = (time.perf_counter() - t0) * 1000
        sources = sorted({d.metadata.get("file_name") for d in docs if d.metadata.get("file_name")})
        yield sse("sources", {"sources": sources})
        first_ms = None
        if not docs:
            yield sse("token", {"text": NOT_FOUND_ANSWER})
        else:
            async for text in astream_answer(query, docs):
                if first_ms is None:
                    first_ms = (time.perf_counter() - t0) * 1000
                yield sse("token", {"text": text})
        yield sse("done", {})
        logger.info("query retrieve_ms=%.0f first_token_ms=%s total_ms=%.0f", retrieve_ms,
                    f"{first_ms:.0f}" if first_ms is not None else "-", (time.perf_counter() - t0) * 1000)
    except Exception:
        logger.exception("streaming query failed")
        yield sse("error", {"detail": "Failed to generate the answer"})


# ---------------------------------------------------------------------------- publishing
def generate_and_store_database_task():
    """Build the index from books/*.pdf and publish it safely (VERSION.json uploaded last)."""
    from PyPDF2 import PdfReader
    from io import BytesIO
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with _update_lock:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        documents = []
        for name in sb.list_files("books"):
            if not name.lower().endswith(".pdf"):
                continue
            try:
                reader = PdfReader(BytesIO(sb.download_bytes(name)))
                # extract_text() may return None for image-only pages
                text = " ".join((page.extract_text() or "") for page in reader.pages)
            except Exception as e:
                logger.warning("Skipping %s: %s", name, e)
                continue
            if text.strip():
                documents.extend(Document(page_content=c, metadata={"file_name": name})
                                 for c in splitter.split_text(text))
        if not documents:
            logger.warning("No PDF text found; database not regenerated.")
            return

        parent = os.path.dirname(os.path.abspath(CHROMA_DIR)) or "."
        os.makedirs(parent, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=".chroma_new_", dir=parent)
        try:
            vs = Chroma.from_documents(documents=documents, embedding=get_embeddings_model(),
                                       persist_directory=tmp)
            try:
                vs._client._system.stop()  # release file handles before uploading
            except Exception:
                pass
            _clear_chroma_cache()

            version = {
                "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "files": sorted({d.metadata["file_name"] for d in documents}),
                "chunks": len(documents),
                "embedding_model": MODEL_NAME,
            }
            with open(_version_path(tmp), "w") as f:
                json.dump(version, f, indent=2)

            # Upload everything except VERSION.json first, VERSION.json LAST.
            rel_paths = []
            for root, _, files in os.walk(tmp):
                for file in files:
                    rel_paths.append(os.path.relpath(os.path.join(root, file), tmp).replace(os.sep, "/"))
            rel_paths.sort(key=lambda p: p == VERSION_FILE)
            for rel in rel_paths:
                sb.upload_file(f"{REMOTE_PREFIX}/{rel}", os.path.join(tmp, *rel.split("/")))
                logger.info("Uploaded %s", rel)

            # Delete stale remote files only after the new upload fully succeeded.
            keep = {f"{REMOTE_PREFIX}/{p}" for p in rel_paths}
            sb.delete_files([p for p in sb.list_files(REMOTE_PREFIX) if p not in keep])
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        _activate(tmp)  # swap local dir + in-memory chain, no restart needed
        logger.info("Published index version %s (%d chunks)", version["version"], len(documents))
