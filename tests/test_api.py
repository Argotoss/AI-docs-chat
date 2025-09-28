import pytest
from fastapi.testclient import TestClient
from app.main import api
import fitz
import io
import os
import tempfile

client = TestClient(api)

# Create a sample PDF with known content
def create_sample_pdf():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is page 1. It contains information about AI.")
    page = doc.new_page()
    page.insert_text((50, 50), "Page 2 discusses machine learning models.")
    bio = io.BytesIO()
    doc.save(bio)
    doc.close()
    bio.seek(0)
    return bio.read()

sample_pdf_bytes = create_sample_pdf()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "api" in data
    assert data["api"] is True

def test_upload_pdf():
    files = {"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert "kb" in data
    doc_id = data["doc_id"]
    return doc_id

def test_ask_question():
    doc_id = test_upload_pdf()  # Upload first
    payload = {"doc_id": doc_id, "question": "What is discussed on page 1?"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert len(data["citations"]) > 0

def test_list_documents():
    test_upload_pdf()  # Ensure at least one doc
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert len(data["documents"]) >= 1

def test_get_document_details():
    doc_id = test_upload_pdf()
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == doc_id

def test_delete_document():
    doc_id = test_upload_pdf()
    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 200
    # Check it's gone
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 404
