import os
import time
from typing import Dict, List, Iterable, Tuple
import numpy as np
import fitz
import json

DATA_DIR = os.path.join(os.getcwd(), "data")

def ensure_doc_dir(doc_id: str) -> str:
    """Create (if needed) and return absolute path for the document directory."""
    d = os.path.join(DATA_DIR, doc_id)
    os.makedirs(d, exist_ok=True)
    return d

def new_doc_id(filename: str) -> str:
    """Return a millisecond timestamp prefixed identifier for a filename."""
    ts = int(time.time() * 1000)
    safe = filename.replace(" ", "_")
    return f"{ts}_{safe}"

def save_source_pdf(doc_id: str, content_bytes: bytes) -> str:
    """Persist raw uploaded PDF bytes for a document and return path."""
    d = ensure_doc_dir(doc_id)
    src_path = os.path.join(d, "source.pdf")
    with open(src_path, "wb") as f:
        f.write(content_bytes)
    return src_path

def save_meta(doc_id: str, filename: str, pdf_path: str) -> Dict:
    """Write metadata file for the document and return the metadata dict."""
    meta = {
        "doc_id": doc_id,
        "filename": filename,
        "pdf_path": pdf_path,
        "created_at": int(time.time()),
    }
    with open(os.path.join(DATA_DIR, doc_id, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


DEFAULT_CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class ProcessingError(Exception):
    """Raised when any stage of the processing pipeline fails."""


def extract_pdf_pages(pdf_path: str) -> List[Dict]:
    """Extract raw text per page using PyMuPDF.

    Returns list of dicts: {page: int, text: str}
    """
    pages: List[Dict] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ProcessingError(f"Could not open PDF: {e}")
    for i, page in enumerate(doc):
        try:
            text = page.get_text("text") or ""
        except Exception:
            text = ""
        pages.append({"page": i, "text": text})
    doc.close()
    return pages


def _yield_page_chunks(page_text: str, page_number: int, chunk_chars: int, overlap: int) -> Iterable[Tuple[int, int, str]]:
    """Yield (start, end, text) slices for a single page respecting overlap."""
    text = page_text.strip()
    if not text:
        return []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            yield start, end, chunk
        if end >= n:
            break
        start = max(0, end - overlap)


def build_chunks(doc_id: str, pages: List[Dict], chunk_chars: int = DEFAULT_CHUNK_CHARS, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict]:
    """Return structured overlapping text chunks for all pages."""
    chunks: List[Dict] = []
    idx = 0
    for p in pages:
        page_num = p["page"]
        for start, end, chunk_text in _yield_page_chunks(p["text"], page_num, chunk_chars, overlap):
            chunks.append({
                "id": f"{doc_id}:{idx}",
                "doc_id": doc_id,
                "index": idx,
                "page": page_num,
                "start": start,
                "end": end,
                "text": chunk_text,
            })
            idx += 1
    return chunks


def embed_chunks(chunks: List[Dict], model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL, timeout: float = 60.0) -> List[List[float]]:
    """Return embedding vectors for chunks using the Ollama embeddings API."""
    import requests
    embeddings: List[List[float]] = []
    url = f"{base_url.rstrip('/')}/api/embeddings"
    for c in chunks:
        payload = {"model": model, "prompt": c["text"]}
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if not r.ok:
                raise ProcessingError(f"Embedding request failed HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            emb = data.get("embedding")
            if not emb:
                raise ProcessingError("No 'embedding' in response")
            embeddings.append(emb)
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Embedding error: {e}")
    return embeddings


def save_chunks_and_embeddings(doc_id: str, chunks: List[Dict], embeddings: List[List[float]]):
    """Persist chunks (jsonl) and embeddings (npy) and return summary info."""
    d = ensure_doc_dir(doc_id)
    chunks_path = os.path.join(d, "chunks.jsonl")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    arr = np.array(embeddings, dtype="float32")
    np.save(os.path.join(d, "embeddings.npy"), arr)
    return {
        "chunks_path": chunks_path,
        "embeddings_path": os.path.join(d, "embeddings.npy"),
        "chunk_count": len(chunks),
        "embedding_dim": 0 if arr.size == 0 else arr.shape[1],
    }


def build_knowledge_base(doc_id: str, pdf_path: str, *, chunk_chars: int = DEFAULT_CHUNK_CHARS, overlap: int = DEFAULT_CHUNK_OVERLAP) -> Dict:
    """Run the full pipeline and return persisted artifact summary."""
    pages = extract_pdf_pages(pdf_path)
    chunks = build_chunks(doc_id, pages, chunk_chars=chunk_chars, overlap=overlap)
    if not chunks:
        return {"chunk_count": 0, "embedding_dim": 0}
    embeddings = embed_chunks(chunks)
    saved = save_chunks_and_embeddings(doc_id, chunks, embeddings)
    return saved

