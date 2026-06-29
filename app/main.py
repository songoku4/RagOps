from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
from app.rag import ingest_pdf, query_rag, setup_mlflow
import time

# Custom metrics
QUERIES_TOTAL = Counter(
    "ragops_queries_total",
    "Total number of queries made"
)
QUERY_LATENCY = Histogram(
    "ragops_query_latency_seconds",
    "Query latency in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)
CHUNKS_RETRIEVED = Histogram(
    "ragops_chunks_retrieved",
    "Number of chunks retrieved per query",
    buckets=[1, 2, 3, 4, 5, 10]
)
DOCUMENTS_INGESTED = Counter(
    "ragops_documents_ingested_total",
    "Total number of documents ingested"
)
CHUNKS_STORED = Gauge(
    "ragops_chunks_stored_total",
    "Total chunks currently stored in ChromaDB"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_mlflow()
    yield

app = FastAPI(title="RAG-Ops Document Intelligence", lifespan=lifespan)

Instrumentator().instrument(app).expose(app)

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
    DOCUMENTS_INGESTED.inc()
    CHUNKS_STORED.set(chunks)
    return {"message": f"Ingested {chunks} chunks from {file.filename}"}

@app.post("/query")
async def query(req: QueryRequest):
    start = time.time()
    result = query_rag(req.question)
    duration = time.time() - start

    QUERIES_TOTAL.inc()
    QUERY_LATENCY.observe(duration)
    CHUNKS_RETRIEVED.observe(len(result.get("sources", [])))

    result["latency_ms"] = round(duration * 1000)
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}