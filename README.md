# AI-doc-chat

FastAPI service for uploading PDFs, processing them into knowledge bases (text extraction, chunking, embeddings), and answering questions with retrieval-augmented generation. Supports document management, quality guardrails, and optional OCR for scanned PDFs.

## Quick Start
```bash
docker-compose up --build
```
Starts Ollama, pulls `nomic-embed-text` and `llama3.1:8b` models, and launches the API on port 8000.

## Endpoints
- `GET /health` - Check API and Ollama status.
- `POST /upload` - Upload PDF, process into KB, return metadata.
- `POST /ask` - Query KB: `{"doc_id": "...", "question": "..."}` → `{"answer": "...", "citations": [...]}` or refusal if low relevance.
- `GET /documents` - List all documents with metadata and chunk counts.
- `GET /documents/{doc_id}` - Get detailed info for a document.
- `DELETE /documents/{doc_id}` - Delete a document and all files.

## Upload & Processing Flow
1. POST `/upload` with PDF file.
2. File saved to `data/<doc_id>/source.pdf`.
3. Metadata saved to `meta.json`.
4. Text extracted per page (PyMuPDF); OCR fallback if enabled and page has no text.
5. Text chunked into overlapping windows (filtered for min length).
6. Chunks embedded via Ollama (`nomic-embed-text`).
7. Chunks saved to `chunks.jsonl`; embeddings to `embeddings.npy`.

## Q&A Flow
1. Embed question with `nomic-embed-text`.
2. Compute cosine similarity to stored embeddings.
3. Retrieve top-k chunks (if similarity > threshold).
4. Generate answer with `llama3.1:8b` (concise, cited, deterministic).
5. Return answer + citations, or "Not enough relevant information" if low similarity.

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHAT_MODEL` | `llama3.1:8b` | Chat model for answers |
| `CHUNK_CHARS` | `1000` | Max chars per chunk |
| `CHUNK_OVERLAP` | `150` | Chunk overlap |
| `TOP_K` | `5` | Top chunks for retrieval |
| `MAX_FILE_MB` | `50` | Max upload size (MB) |
| `SIMILARITY_THRESHOLD` | `0.3` | Min similarity to answer |
| `MAX_ANSWER_LENGTH` | `500` | Max answer words |
| `MIN_CHUNK_LENGTH` | `100` | Min chunk chars (filter noise) |
| `ENABLE_OCR` | `false` | Enable OCR fallback for image PDFs |

## Data Artifacts per Document
```
data/<doc_id>/
	source.pdf
	meta.json
	chunks.jsonl        # {id, doc_id, index, page, start, end, text}
	embeddings.npy      # float32 matrix (num_chunks x dim)
```

