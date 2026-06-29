import os
import time
import tempfile
import mlflow
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "./models/all-MiniLM-L6-v2")
LLM_MODEL = "llama3.2"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

def setup_mlflow():
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("ragops")
    except Exception as e:
        print(f"[MLFLOW] Setup failed: {e}")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, say "I could not find that in the document."

Context:
{context}

Question: {question}

Answer:
""")

def get_db():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return Chroma(
        client=client,
        collection_name="ragops",
        embedding_function=embeddings
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    print(f"[INGEST] Received {len(file_bytes)} bytes for {filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode='wb') as f:
        f.write(file_bytes)
        tmp_path = f.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        print(f"[INGEST] Loaded {len(docs)} pages")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        print(f"[INGEST] Created {len(chunks)} chunks")

        if not chunks:
            return 0

        db = get_db()
        db.add_documents(chunks)
        print(f"[INGEST] Stored {len(chunks)} chunks")
        try:
            with mlflow.start_run(run_name=f"ingest_{filename}"):
                mlflow.log_param("filename", filename)
                mlflow.log_param("chunk_size", CHUNK_SIZE)
                mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
                mlflow.log_param("embedding_model", EMBEDDING_MODEL)
                mlflow.log_metric("pages_loaded", len(docs))
                mlflow.log_metric("chunks_created", len(chunks))
        except Exception as e:
            print(f"[MLFLOW] Logging failed: {e}")

        return len(chunks)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def query_rag(question: str) -> dict:
    db = get_db()
    count = db._collection.count()
    print(f"[QUERY] ChromaDB has {count} chunks")

    if count == 0:
        return {
            "answer": "No documents ingested yet. Please upload a PDF first.",
            "sources": []
        }

    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_HOST)

    start = time.time()
    retrieved_docs = retriever.invoke(question)
    retrieval_time = round((time.time() - start) * 1000)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    llm_start = time.time()
    answer = chain.invoke(question)
    llm_time = round((time.time() - llm_start) * 1000)
    total_time = round((time.time() - start) * 1000)

    try:
        with mlflow.start_run(run_name="query"):
            mlflow.log_param("question", question[:100])
            mlflow.log_param("llm_model", LLM_MODEL)
            mlflow.log_param("embedding_model", EMBEDDING_MODEL)
            mlflow.log_metric("retrieval_latency_ms", retrieval_time)
            mlflow.log_metric("llm_latency_ms", llm_time)
            mlflow.log_metric("total_latency_ms", total_time)
            mlflow.log_metric("chunks_retrieved", len(retrieved_docs))
    except Exception as e:
        print(f"[MLFLOW] Logging failed: {e}")

    print(f"[QUERY] Retrieved {len(retrieved_docs)} chunks in {retrieval_time}ms, LLM in {llm_time}ms")

    sources = []
    for doc in retrieved_docs:
        sources.append({
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:150]
        })

    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": total_time
    }