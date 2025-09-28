import pytest
from fastapi.testclient import TestClient
from app.main import api
import fitz
import io
import os
import json

client = TestClient(api)

# Create a sample PDF with known content
def create_sample_pdf():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is page 1. It contains information about AI and machine learning models. This is a longer text to ensure it meets the minimum chunk length requirement for testing purposes. We need at least 100 characters here to create valid chunks.")
    page = doc.new_page()
    page.insert_text((50, 50), "Page 2 discusses machine learning models in detail. This page also has enough content to create chunks when processed by the chunking function. Let's add more words to make it sufficiently long for the tests.")
    bio = io.BytesIO()
    doc.save(bio)
    doc.close()
    bio.seek(0)
    return bio.read()

sample_pdf_bytes = create_sample_pdf()

@pytest.fixture
def mock_apis(mocker):
    def mock_post(url, json=None, **kwargs):
        mock_response = mocker.Mock()
        mock_response.ok = True
        if json and "messages" in json:
            # Chat request
            mock_response.json.return_value = {"message": {"content": "This is a test answer about AI and machine learning."}}
        else:
            # Embedding request
            mock_response.json.return_value = {"embedding": [0.1] * 768}
        return mock_response
    mocker.patch("requests.post", side_effect=mock_post)

@pytest.fixture
def uploaded_doc_id(mock_apis):
    files = {"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    return data["doc_id"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "api" in data
    assert data["api"] is True

def test_upload_pdf(mock_apis):
    files = {"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert "kb" in data

def test_ask_question(mock_apis, uploaded_doc_id):
    payload = {"doc_id": uploaded_doc_id, "question": "What is discussed on page 1?"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert len(data["citations"]) > 0

def test_list_documents(mock_apis, uploaded_doc_id):
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert len(data["documents"]) >= 1

def test_get_document_details(mock_apis, uploaded_doc_id):
    response = client.get(f"/documents/{uploaded_doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == uploaded_doc_id

def test_delete_document(mock_apis, uploaded_doc_id):
    response = client.delete(f"/documents/{uploaded_doc_id}")
    assert response.status_code == 200
    # Check it's gone
    response = client.get(f"/documents/{uploaded_doc_id}")
    assert response.status_code == 404
