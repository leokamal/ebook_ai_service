from fastapi import FastAPI, BackgroundTasks
from firebase_admin import credentials, storage
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from io import BytesIO
import tempfile
import os
import firebase_admin
from langchain.schema import Document

package__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

app = FastAPI()

bucket_name = "ebook-ai-e51a0.appspot.com"

# Check if an app is already initialized
if not firebase_admin._apps:
    # Initialize Firebase Admin SDK if no app exists
    #cred = credentials.Certificate("config/ebook-ai-e51a0-firebase-adminsdk.json")
    cred = credentials.Certificate("/etc/secrets/ebook-ai-e51a0-firebase-adminsdk.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': bucket_name  # Replace with your actual Firebase project ID
    })
else:
    print("Firebase app already initialized.")

#Add Google_api_key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCcZb8kY7KsefGijTiumFfFRB-W8FZc-6A"

#Generate and Store Database
def generate_and_store_database_task():
    storage_bucket = storage.bucket(bucket_name)
    blobs = storage_bucket.list_blobs(prefix="books/")
    
    documents = []
    for blob in blobs:
        if blob.name.endswith('.pdf'):
            pdf_content = PdfReader(BytesIO(blob.download_as_bytes()))
            text = ' '.join(page.extract_text() for page in pdf_content.pages)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split_text(text)
            documents.extend([Document(page_content=chunk, metadata={"file_name": blob.name}) for chunk in chunks])
    model_name = "sentence-transformers/all-MiniLM-L6-v2";
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    # Save the Chroma database to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=documents, 
            embedding=embeddings_model,
            persist_directory=temp_dir
        )
        vector_store.persist()

        # Upload the database files to Firebase Storage
        for root, _, files in os.walk(temp_dir):
            for file in files:
                blob = storage_bucket.blob(f"chroma_db/{file}")
                blob.upload_from_filename(os.path.join(root, file))
                print(f"Uploaded {file} to Firebase Storage")

def load_database_from_firebase(temp_dir):
    storage_bucket = storage.bucket(bucket_name)
    blobs = storage_bucket.list_blobs(prefix="chroma_db/")
    
    # Download the database files from Firebase Storage
    for blob in blobs:
        local_path = os.path.join(temp_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")

def query_database(query):
    with tempfile.TemporaryDirectory() as temp_dir:
        load_database_from_firebase(temp_dir)

        model_name = "sentence-transformers/all-MiniLM-L6-v2";
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        vector_store = Chroma(embedding_function=embeddings_model, persist_directory=temp_dir)
        prompt = PromptTemplate.from_template(
            """ 
                You are a highly skilled assistant who excels in rewriting and summarizing information based on a given query. Given the following context and question, your task is to rewrite the context so that it directly addresses the question. Ensure that the rewritten context is relevant, concise, and directly related to the user's query.

            **Context:** {context}

            **Question:** {input}

            **Rewritten Context:**
        """
        )
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.4})
        model_name="gemini-pro"
        llm = ChatGoogleGenerativeAI(model=model_name)
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
        return chain.invoke({"input": query})



# Start FastAPI with: uvicorn script_name:app --reload
