import pytest
import os
import tempfile
import shutil
from app.storage import (
    build_chunks, embed_chunks, compute_cosine_similarity,
    get_top_k_chunks, extract_pdf_pages, _yield_page_chunks
)
import numpy as np

# Mock data
sample_pages = [
    {"page": 0, "text": "This is the first page with some content. It has multiple sentences."},
    {"page": 1, "text": "Second page here. More text for testing purposes."}
]

def test_yield_page_chunks():
    page_text = "This is a test. It should split into chunks."
    chunks = list(_yield_page_chunks(page_text, 0, chunk_chars=20, overlap=5))
    assert len(chunks) > 0
    assert all(len(chunk[2]) <= 20 for chunk in chunks)

def test_build_chunks():
    chunks = build_chunks("test_doc", sample_pages, chunk_chars=50, overlap=10)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "id" in chunk
        assert "text" in chunk
        assert len(chunk["text"]) >= 10  # min length

def test_compute_cosine_similarity():
    query_emb = [1.0, 0.0]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    similarities = compute_cosine_similarity(query_emb, embeddings)
    assert similarities[0] == pytest.approx(1.0)
    assert similarities[1] == pytest.approx(0.0)

def test_get_top_k_chunks():
    chunks = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    similarities = np.array([0.5, 0.9, 0.1])
    top = get_top_k_chunks(chunks, similarities, k=2)
    assert len(top) == 2
    assert top[0]["id"] == "2"  # highest similarity

# Note: embed_chunks requires Ollama, so skip or mock in real tests
# For now, assume it's tested indirectly in E2E
