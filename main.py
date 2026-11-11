# Workaround to ensure `pysqlite3` is used as `sqlite3` (must run before chromadb is imported)
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import os
import warnings
import uvicorn
from rag_service import query_database, generate_and_store_database_task, load_database_from_firebase, get_query_chain
from ocr_service import perform_ocr  # Import the updated OCR service

MAX_UPLOAD_SIZE = 10 * 1024 * 1024

# Ignore warnings from pytesseract
warnings.filterwarnings("ignore", category=UserWarning, module='easyocr')

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for your specific needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bucket name configuration
bucket_name = "ebook-ai-e51a0.appspot.com"

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

# Query the database endpoint
@app.get("/query-database/")
def query_database_endpoint(query: str = Query(...)):
    try:
        response = query_database(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"response": response}

# Test endpoint
@app.get("/test/")
def test():
    return {"response": "Hello world"}

# Generate and store the database endpoint
@app.post("/generate-database/")
def generate_database(background_tasks: BackgroundTasks):
    background_tasks.add_task(generate_and_store_database_task)
    return {"message": "Database generation and storage started"}

@app.post("/load-database/")
def load_database(background_tasks: BackgroundTasks):
    # Define a persistent local directory for the database files
    local_dir = "./chroma_db_local"
    
    # Create the directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Re-download the latest database and rebuild the cached query chain
    load_database_from_firebase(local_dir, force=True)
    get_query_chain.cache_clear()
    return {"message": "Database loaded successfully"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
