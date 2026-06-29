import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

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

    print(f"[INGEST] Temp file size: {os.path.getsize(tmp_path)} bytes")

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        print(f"[INGEST] Loaded {len(docs)} pages")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)
        print(f"[INGEST] Created {len(chunks)} chunks")

        if not chunks:
            print("[INGEST] ERROR: No chunks created")
            return 0

        db = get_db()
        db.add_documents(chunks)
        print(f"[INGEST] Stored {len(chunks)} chunks")
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

    retrieved_docs = retriever.invoke(question)
    print(f"[QUERY] Retrieved {len(retrieved_docs)} chunks")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    sources = []
    for doc in retrieved_docs:
        sources.append({
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:150]
        })

    return {
        "answer": answer,
        "sources": sources
    }