FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    pypdf \
    langchain \
    langchain-core \
    langchain-community \
    langchain-text-splitters \
    langchain-huggingface \
    langchain-ollama \
    langchain-chroma \
    chromadb \
    sentence-transformers

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]