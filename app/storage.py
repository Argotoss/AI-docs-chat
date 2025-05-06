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
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")
TOP_K = int(os.getenv("TOP_K", "5"))


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


# ---------------- Retrieval & Q&A ---------------- #

def load_chunks(doc_id: str) -> List[Dict]:
    """Load chunks from jsonl file."""
    path = os.path.join(DATA_DIR, doc_id, "chunks.jsonl")
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_embeddings(doc_id: str) -> np.ndarray:
    """Load embeddings from npy file."""
    path = os.path.join(DATA_DIR, doc_id, "embeddings.npy")
    return np.load(path)


def embed_query(query: str, model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL, timeout: float = 60.0) -> List[float]:
    """Embed a single query string."""
    import requests
    url = f"{base_url.rstrip('/')}/api/embeddings"
    payload = {"model": model, "prompt": query}
    r = requests.post(url, json=payload, timeout=timeout)
    if not r.ok:
        raise ProcessingError(f"Query embedding failed HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    emb = data.get("embedding")
    if not emb:
        raise ProcessingError("No 'embedding' in query response")
    return emb


def compute_cosine_similarity(query_emb: List[float], embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all embeddings."""
    query_vec = np.array(query_emb)
    norms_q = np.linalg.norm(query_vec)
    norms_e = np.linalg.norm(embeddings, axis=1)
    dot_products = np.dot(embeddings, query_vec)
    similarities = dot_products / (norms_e * norms_q)
    return similarities


def get_top_k_chunks(chunks: List[Dict], similarities: np.ndarray, k: int = TOP_K) -> List[Dict]:
    """Return top-k chunks by similarity."""
    top_indices = np.argsort(similarities)[::-1][:k]
    return [chunks[i] for i in top_indices]


def generate_answer(context: str, question: str, model: str = CHAT_MODEL, base_url: str = OLLAMA_BASE_URL, timeout: float = 300.0) -> str:
    """Generate answer using chat model with context."""
    import requests
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely with citations to page numbers if relevant."
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=timeout)
    if not r.ok:
        raise ProcessingError(f"Chat failed HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    answer = data.get("message", {}).get("content", "")
    if not answer:
        raise ProcessingError("No answer in chat response")
    return answer


def ask_question(doc_id: str, question: str, k: int = TOP_K) -> Dict:
    """Full retrieval + Q&A pipeline."""
    chunks = load_chunks(doc_id)
    embeddings = load_embeddings(doc_id)
    query_emb = embed_query(question)
    similarities = compute_cosine_similarity(query_emb, embeddings)
    top_chunks = get_top_k_chunks(chunks, similarities, k)
    context = "\n\n".join([f"Page {c['page']}: {c['text']}" for c in top_chunks])
    answer = generate_answer(context, question)
    citations = [{"page": c["page"], "chunk_id": c["id"]} for c in top_chunks]
    return {"answer": answer, "citations": citations}

