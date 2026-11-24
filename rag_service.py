import supabase_storage as sb
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from io import BytesIO
import tempfile
import os
import json
import shutil
from datetime import datetime, timezone
from langchain_core.documents import Document
from functools import lru_cache

# NOTE: heavy imports (torch/sentence-transformers, chroma, gemini) are done lazily
# inside the functions below so the web server can bind its port quickly on startup.
# The pysqlite3 -> sqlite3 swap is done in main.py before anything imports chromadb.

# Storage is Supabase (see supabase_storage.py). GOOGLE_API_KEY must be set in the environment.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIR = "./chroma_db_local"
VERSION_FILE = "VERSION.json"


@lru_cache(maxsize=1)
def get_embeddings_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


@lru_cache(maxsize=1)
def get_query_chain():
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_google_genai import ChatGoogleGenerativeAI
    os.makedirs(LOCAL_DIR, exist_ok=True)
    try:
        load_database_from_supabase(LOCAL_DIR, force=True)
    except Exception as e:
        # e.g. Supabase unreachable -> fall back to a previously downloaded local copy
        if not os.path.exists(os.path.join(LOCAL_DIR, "chroma.sqlite3")):
            raise
        print(f"Could not refresh database from Supabase ({e}); using local copy.")
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
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
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
    from langchain_chroma import Chroma
    # List all files under "books/" in the Supabase bucket
    book_paths = sb.list_files("books")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

    documents = []
    for name in book_paths:
        if name.lower().endswith('.pdf'):
            try:
                pdf_bytes = sb.download_bytes(name)
                pdf_content = PdfReader(BytesIO(pdf_bytes))
                # extract_text() may return None for image-only pages
                text = ' '.join((page.extract_text() or '') for page in pdf_content.pages)
            except Exception as e:
                print(f"Skipping {name}: {e}")
                continue
            if not text.strip():
                continue
            chunks = splitter.split_text(text)
            documents.extend([Document(page_content=chunk, metadata={"file_name": name}) for chunk in chunks])

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

        # Version marker, stored identically in the local copy and in Supabase
        version = {
            "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "files": sorted(set(d.metadata["file_name"] for d in documents)),
            "chunks": len(documents),
            "embedding_model": MODEL_NAME,
        }
        with open(os.path.join(temp_dir, VERSION_FILE), "w") as f:
            json.dump(version, f, indent=2)

        # Keep a local copy (./chroma_db_local) in sync with what is uploaded
        if os.path.isdir(LOCAL_DIR):
            shutil.rmtree(LOCAL_DIR, ignore_errors=True)
        shutil.copytree(temp_dir, LOCAL_DIR, dirs_exist_ok=True)
        print(f"Local database written to {LOCAL_DIR} (version {version['version']})")

        # Remove stale files from the previous database
        sb.delete_files(sb.list_files("chroma_db"))

        # Upload the database files, preserving the directory structure
        for root, _, files in os.walk(temp_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, temp_dir).replace(os.sep, "/")
                sb.upload_file(f"chroma_db/{rel_path}", full_path)
                print(f"Uploaded {rel_path} to Supabase Storage")

    get_query_chain.cache_clear()

# Function to load the database from Supabase Storage
def load_database_from_supabase(local_dir, force=False):
    prefix = "chroma_db/"
    names = sb.list_files("chroma_db")

    base = os.path.realpath(local_dir)
    for name in names:
        rel_path = name[len(prefix):]
        if not rel_path:
            continue
        parts = rel_path.split("/")
        if any(p in ("", ".", "..") for p in parts):
            print(f"Skipping suspicious object name {name}")
            continue
        local_path = os.path.realpath(os.path.join(base, *parts))
        if os.path.commonpath([base, local_path]) != base:
            print(f"Skipping object outside target dir: {name}")
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path) or force:
            tmp_path = local_path + ".part"
            try:
                with open(tmp_path, "wb") as f:
                    f.write(sb.download_bytes(name))
                os.replace(tmp_path, local_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            print(f"Downloaded {name} to {local_path}")
        else:
            print(f"File {local_path} already exists, skipping download.")


# Function to query the database
def query_database(query):
    result = get_query_chain().invoke({"input": query})
    return result["answer"]


# To start FastAPI, use: uvicorn script_name:app --reload
