from fastapi import FastAPI, Query, BackgroundTasks
from rag_service import  query_database, generate_and_store_database_task

app = FastAPI()

bucket_name = "ebook-ai-e51a0.appspot.com"

@app.get("/query-database/")
def query_database_endpoint( query: str = Query(...)) :
    response = query_database(query)
    return {"response": response}

@app.get("/test/")
def test():
    return {"response": "Hello world"}
    
@app.post("/generate-database/")
def generate_database(background_tasks: BackgroundTasks):
    background_tasks.add_task(generate_and_store_database_task)
    return {"message": "Database generation and storage started"}