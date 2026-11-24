# Workaround to ensure `pysqlite3` is used as `sqlite3` (must run before chromadb is imported)
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()  # load SUPABASE_* / GOOGLE_API_KEY from a local .env (git-ignored)
except ImportError:
    pass

import asyncio
import os
import warnings
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import rag_service
from ocr_service import perform_ocr  # Import the updated OCR service
from security import verify_admin_token

import logging
logger = logging.getLogger("uvicorn.error")

MAX_UPLOAD_SIZE = 10 * 1024 * 1024

# Ignore warnings from pytesseract
warnings.filterwarnings("ignore", category=UserWarning, module='easyocr')

@asynccontextmanager
async def lifespan(_app):
    # Warm up in a background thread so the port binds immediately.
    rag_service.start_warmup_thread()
    yield


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    file_name: Optional[str] = None


async def require_ready():
    if not await asyncio.to_thread(rag_service.wait_ready, 30):
        status = rag_service.get_health()
        if status["status"] == "error":
            raise HTTPException(status_code=503, detail=f"Index unavailable: {status.get('error')}")
        raise HTTPException(status_code=503, detail="Service warming up, retry shortly")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for your specific needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hello World endpoint
@app.get("/hello")
async def hello_world():
    return {"message": "Hello, World!"}

# Upload and process file endpoint using the OCR service
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), lang: str = Query(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    file_bytes = await file.read(MAX_UPLOAD_SIZE + 1)
    if len(file_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        # OCR is CPU-bound/blocking: run it off the event loop
        extracted_text = await run_in_threadpool(perform_ocr, file_bytes, lang)
        return {"extracted_text": extracted_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return rag_service.get_health()


# Streaming query endpoint (Server-Sent Events) - see docs/API.md
@app.post("/query", dependencies=[Depends(require_ready)])
async def query_stream(body: QueryRequest):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")
    return StreamingResponse(
        rag_service.stream_query_events(query, body.file_name),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Backward-compatible non-streaming endpoint
@app.get("/query-database/", dependencies=[Depends(require_ready)])
def query_database_endpoint(query: str = Query(..., min_length=1, max_length=1000)):
    try:
        return rag_service.query_database(query)  # {"answer": ..., "context": [...]}
    except Exception as e:
        rag_service.logger.exception("query-database failed")
        raise HTTPException(status_code=500, detail=str(e))


# Test endpoint
@app.get("/test/")
def test():
    return {"response": "Hello world"}


# Admin endpoints (require X-Admin-Token == $ADMIN_TOKEN)
@app.post("/generate-database/", dependencies=[Depends(verify_admin_token)])
def generate_database(background_tasks: BackgroundTasks):
    background_tasks.add_task(rag_service.generate_and_store_database_task)
    return {"message": "Database generation and storage started"}


@app.post("/load-database/", dependencies=[Depends(verify_admin_token)])
def load_database():
    # Downloads only if the remote VERSION.json differs, then swaps the in-memory chain.
    changed = rag_service.ensure_local_index()
    return {"message": "Database loaded successfully" if changed else "Already up to date", "changed": changed}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
