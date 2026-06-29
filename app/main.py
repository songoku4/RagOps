from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.rag import ingest_pdf, query_rag
import time

app = FastAPI(title="RAG-Ops Document Intelligence")

class QueryRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("app/templates/index.html") as f:
        return f.read()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    contents = await file.read()
    chunks = ingest_pdf(contents, file.filename)
    return {"message": f"Ingested {chunks} chunks from {file.filename}"}

@app.post("/query")
async def query(req: QueryRequest):
    start = time.time()
    result = query_rag(req.question)
    result["latency_ms"] = round((time.time() - start) * 1000)
    return result