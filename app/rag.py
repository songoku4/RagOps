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

CHROMA_PATH = "chroma_db"
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

def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    print(f"[INGEST] Received {len(file_bytes)} bytes for {filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode='wb') as f:
        f.write(file_bytes)
        tmp_path = f.name

    print(f"[INGEST] Temp file: {tmp_path}, size: {os.path.getsize(tmp_path)} bytes")

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
            print("[INGEST] ERROR: No chunks — PDF may be image-based or empty")
            return 0

        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        db.add_documents(chunks)
        print(f"[INGEST] Stored {len(chunks)} chunks in ChromaDB")
        return len(chunks)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def query_rag(question: str) -> dict:
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    count = db._collection.count()
    print(f"[QUERY] ChromaDB has {count} chunks")

    if count == 0:
        return {
            "answer": "No documents ingested yet. Please upload a PDF first.",
            "sources": []
        }

    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model=LLM_MODEL)

    retrieved_docs = retriever.invoke(question)
    print(f"[QUERY] Retrieved {len(retrieved_docs)} chunks")
    for i, doc in enumerate(retrieved_docs):
        print(f"[QUERY] Chunk {i}: {doc.page_content[:100]}")

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