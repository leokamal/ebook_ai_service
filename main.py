from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import easyocr
import io
import os
import warnings
from PIL import Image, ImageEnhance, ImageFilter
import sys
from rag_service import query_database, generate_and_store_database_task, load_database_from_firebase
from ocr_service import perform_ocr  # Import the updated OCR service

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

# Workaround to ensure `pysqlite3` is used as `sqlite3`
pysqlite3 = __import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Hello World endpoint
@app.get("/hello")
async def hello_world():
    return {"message": "Hello, World!"}

# Upload and process file endpoint using the OCR service
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), lang: str = Query(...)):
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        file_bytes = await file.read()
        extracted_text = perform_ocr(file_bytes, lang)  # Pass the language to the OCR function
        return {"extracted_text": extracted_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Query the database endpoint
@app.get("/query-database/")
def query_database_endpoint(query: str = Query(...)):
    response = query_database(query)
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

    # Load the database files from Firebase Storage if not already present locally
    
    load_database_from_firebase(local_dir,True)
    return {"message": "Database loaded with successfully"}

# To run the server, use the command: uvicorn main:app --reload
