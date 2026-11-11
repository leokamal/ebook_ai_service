from firebase_admin import credentials, storage
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from io import BytesIO
import tempfile
import os
import firebase_admin
from langchain_core.documents import Document
from functools import lru_cache


pysqlite3 = __import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Define the name of the Firebase Storage bucket
bucket_name = "ebook-ai-e51a0.appspot.com"

# Initialize Firebase Admin SDK if it's not already initialized
if not firebase_admin._apps:
    # Load Firebase credentials from a JSON file
    # cred = credentials.Certificate("config/ebook-ai-e51a0-firebase-adminsdk.json")
    cred = credentials.Certificate("/etc/secrets/ebook-ai-e51a0-firebase-adminsdk.json")
    # Initialize the Firebase app with the credentials and bucket name
    firebase_admin.initialize_app(cred, {
        'storageBucket': bucket_name  # Replace with your actual Firebase project ID
    })
else:
    print("Firebase app already initialized.")

# Add Google API key to environment variables if not already set
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBLW53i_x5HdRJbYRv9-Vj--fCdv_Urrh8"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIR = "./chroma_db_local"


@lru_cache(maxsize=1)
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


@lru_cache(maxsize=1)
def get_query_chain():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    load_database_from_firebase(LOCAL_DIR, force=True)
    vector_store = Chroma(
        embedding_function=get_embeddings_model(),
        persist_directory=LOCAL_DIR
    )
    prompt = PromptTemplate.from_template(
        """
            You are a highly skilled assistant who excels in rewriting, summarizing, and generating relevant information based on a given context.
            Your task is to rewrite and summarize the context so that it directly addresses the question.
            Ensure that the rewritten context is relevant, concise, and directly related to the user's question.
            If the data (context) doesn't exist, say so.

            **Context:** {context}

            **Question:** {input}

            **Rewritten and Summarized Context:**
        """
    )
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.4}
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    answer_chain = (
        (lambda x: {
            "context": "\n\n".join(d.page_content for d in x["context"]),
            "input": x["input"],
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    # Same output shape as create_retrieval_chain: {"input", "context", "answer"}
    return (
        RunnablePassthrough.assign(context=(lambda x: x["input"]) | retriever)
        .assign(answer=answer_chain)
    )

# Function to generate and store the database of PDF documents
def generate_and_store_database_task():
    # Access the Firebase Storage bucket
    storage_bucket = storage.bucket(bucket_name)
    # List all blobs (files) with a prefix of "books/"
    blobs = storage_bucket.list_blobs(prefix="books/")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

    documents = []
    for blob in blobs:
        if blob.name.lower().endswith('.pdf'):
            try:
                pdf_bytes = blob.download_as_bytes()
                pdf_content = PdfReader(BytesIO(pdf_bytes))
                # extract_text() may return None for image-only pages
                text = ' '.join((page.extract_text() or '') for page in pdf_content.pages)
            except Exception as e:
                print(f"Skipping {blob.name}: {e}")
                continue
            if not text.strip():
                continue
            chunks = splitter.split_text(text)
            documents.extend([Document(page_content=chunk, metadata={"file_name": blob.name}) for chunk in chunks])

    if not documents:
        print("No PDF text found; database not regenerated.")
        return

    embeddings_model = get_embeddings_model()

    # Save the Chroma vector store to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings_model,
            persist_directory=temp_dir
        )
        # Release file handles before uploading (required on some platforms)
        try:
            vector_store._client._system.stop()
        except Exception:
            pass

        # Remove stale files from the previous database
        for old_blob in list(storage_bucket.list_blobs(prefix="chroma_db/")):
            old_blob.delete()

        # Upload the database files, preserving the directory structure
        for root, _, files in os.walk(temp_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, temp_dir).replace(os.sep, "/")
                storage_bucket.blob(f"chroma_db/{rel_path}").upload_from_filename(full_path)
                print(f"Uploaded {rel_path} to Firebase Storage")

    get_query_chain.cache_clear()

# Function to load the database from Firebase Storage
def load_database_from_firebase(local_dir, force=False):
    storage_bucket = storage.bucket(bucket_name)
    prefix = "chroma_db/"
    blobs = storage_bucket.list_blobs(prefix=prefix)

    base = os.path.realpath(local_dir)
    for blob in blobs:
        rel_path = blob.name[len(prefix):]
        if not rel_path or blob.name.endswith("/"):
            continue  # skip "directory" placeholder blobs
        parts = rel_path.split("/")
        if any(p in ("", ".", "..") for p in parts):
            print(f"Skipping suspicious blob name {blob.name}")
            continue
        local_path = os.path.realpath(os.path.join(base, *parts))
        if os.path.commonpath([base, local_path]) != base:
            print(f"Skipping blob outside target dir: {blob.name}")
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path) or force:
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")
        else:
            print(f"File {local_path} already exists, skipping download.")


# Function to query the database
def query_database(query):
    return get_query_chain().invoke({"input": query})


# To start FastAPI, use: uvicorn script_name:app --reload
