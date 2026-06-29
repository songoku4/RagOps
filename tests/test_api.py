import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)

def test_home_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG-Ops" in response.text

def test_upload_rejects_non_pdf():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]

def test_query_no_documents():
    with patch("app.main.query_rag") as mock_query:
        mock_query.return_value = {
            "answer": "No documents ingested yet. Please upload a PDF first.",
            "sources": [],
            "latency_ms": 0
        }
        response = client.post(
            "/query",
            json={"question": "what is this about?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

def test_upload_requires_file():
    response = client.post("/upload")
    assert response.status_code == 422

def test_query_requires_question():
    response = client.post("/query", json={})
    assert response.status_code == 422