from fastapi import FastAPI, BackgroundTasks
from firebase_admin import credentials, storage
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from io import BytesIO
import tempfile
import os
import firebase_admin
from langchain.schema import Document


pysqlite3 = __import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

app = FastAPI()

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
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCcZb8kY7KsefGijTiumFfFRB-W8FZc-6A"

# Function to generate and store the database of PDF documents
def generate_and_store_database_task():
    # Access the Firebase Storage bucket
    storage_bucket = storage.bucket(bucket_name)
    # List all blobs (files) with a prefix of "books/"
    blobs = storage_bucket.list_blobs(prefix="books/")
    
    documents = []
    for blob in blobs:
        if blob.name.endswith('.pdf'):
            # Read the PDF file from Firebase Storage
            pdf_content = PdfReader(BytesIO(blob.download_as_bytes()))
            # Extract text from each page of the PDF
            text = ' '.join(page.extract_text() for page in pdf_content.pages)
            # Split the extracted text into chunks
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split_text(text)
            # Create Document objects for each chunk and add metadata
            documents.extend([Document(page_content=chunk, metadata={"file_name": blob.name}) for chunk in chunks])
    
    # Define the model name for embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    # Save the Chroma vector store to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=documents, 
            embedding=embeddings_model,
            persist_directory=temp_dir
        )
        # Persist the vector store
        #vector_store.persist()

        # Upload the database files to Firebase Storage
        for root, _, files in os.walk(temp_dir):
            for file in files:
                blob = storage_bucket.blob(f"chroma_db/{file}")
                blob.upload_from_filename(os.path.join(root, file))
                print(f"Uploaded {file} to Firebase Storage")

# Function to load the database from Firebase Storage
def load_database_from_firebase(local_dir, force=False):
    storage_bucket = storage.bucket(bucket_name)
    # List all blobs (files) with a prefix of "chroma_db/"
    blobs = storage_bucket.list_blobs(prefix="chroma_db/")
    
    # Download the database files from Firebase Storage
    for blob in blobs:
        local_path = os.path.join(local_dir, os.path.basename(blob.name))
        # Check if the file already exists locally
        if not os.path.exists(local_path) or force:
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")
        else:
            print(f"File {local_path} already exists, skipping download.")
            

# Function to query the database
def query_database(query):
    # Define a persistent local directory for the database files
    local_dir = "./chroma_db_local"
    
    # Create the directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Load the database files from Firebase Storage if not already present locally
    load_database_from_firebase(local_dir)

    # Define the model name for embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    # Initialize the Chroma vector store using the local directory
    vector_store = Chroma(embedding_function=embeddings_model, persist_directory=local_dir)
    
    # Define a prompt template for the query
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
  
    # Create a retriever with similarity score threshold
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.4})
    
    # Initialize the language model for responses
    model_name = "gemini-pro"
    llm = ChatGoogleGenerativeAI(model=model_name)
    
    # Create a retrieval chain to process the query
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    
    # Invoke the chain and return the response
    return chain.invoke({"input": query})


# To start FastAPI, use: uvicorn script_name:app --reload
